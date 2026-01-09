import cv2
import threading
import time
import base64
import logging
import numpy as np
from app import socketio
from app.core.face_recognition import FaceRecognizer
from app.core.posture import PostureAnalyzer
from app.core.gaze import GazeTracker
from app.core.fusion import BehaviorFusion
from app.core.event_logger import EventLogger
from app.core.tracker import PersonTracker
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global lock for thread safety if needed
lock = threading.Lock()

class AttentionDetector:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.is_running = False
        self.thread = None
        self.latest_frame = None
        
        # Modules (Lazy loaded in start())
        self.face_rec = None
        self.posture = None
        self.gaze = None
        self.fusion = None
        self.logger = None
        self.tracker = None
        self.models_loaded = False
        
        # --- Accuracy & Stability Enhancements ---
        self.state_history = {} # {id: [state1, state2, ...]} (buffer size 15)
        self.VOTE_WINDOW = 15
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        # Runtime Data
        self.students_status = {} # {id: {status: '...', name: '...', score: 90}}
        self.last_seen = {} # {id: timestamp}
        self.id_map = {} # {tracker_id: {'roll_no': ..., 'name': ...}}
        self.identity_history = {} # {tracker_id: [roll_no1, roll_no2, ...]}
        self.MAX_IDENTITY_HISTORY = 20
        
        self.ABSENCE_THRESHOLD = 7.0 # Seconds
        self.skip_frames = 2 # Process every Nth frame
        self.frame_count = 0
        
        # Load Balancing / Staggered Analysis
        self.analysis_stagger = 4 # Detailed analysis every N frames per student
        self.student_analysis_count = {} # {id: counter}

    def _load_models(self, app=None):
        """Lazy load heavy AI models only when needed."""
        if not self.models_loaded:
            print("Loading Face Recognizer...")
            self.face_rec = FaceRecognizer()
            print("Loading Posture Analyzer...")
            self.posture = PostureAnalyzer()
            print("Loading Gaze Tracker...")
            self.gaze = GazeTracker()
            print("Loading Fusion Engine...")
            self.fusion = BehaviorFusion()
            print("Loading Event Logger...")
            self.logger = EventLogger(app_context=app)
            print("Initializing Tracker...")
            self.tracker = PersonTracker()
            self.models_loaded = True

    def start(self, app=None):
        if not self.is_running:
            self._load_models(app)
            self.is_running = True
            # Use use_reloader=False in run script to avoid double start
            self.thread = threading.Thread(target=self._process_loop)
            self.thread.daemon = True
            self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join()

    def _preprocess(self, frame):
        """Enhance frame for analysis (Low-light & Noise reduction)."""
        # 1. Noise Reduction
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        
        # 2. Low Light Handling (CLAHE) - apply to L channel in LAB space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_en = self.clahe.apply(l)
        lab_en = cv2.merge((l_en, a, b))
        frame_en = cv2.cvtColor(lab_en, cv2.COLOR_LAB2BGR)
        
        return frame_en

    def _get_voted_state(self, student_id, new_state):
        """Apply multi-frame voting to stabilize state changes."""
        if student_id not in self.state_history:
            self.state_history[student_id] = []
            
        self.state_history[student_id].append(new_state)
        
        # Keep window size
        if len(self.state_history[student_id]) > self.VOTE_WINDOW:
            self.state_history[student_id].pop(0)
            
        # Return most common state (Mode)
        from collections import Counter
        counts = Counter(self.state_history[student_id])
        return counts.most_common(1)[0][0]

    def _process_loop(self):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open video source {self.camera_id}")
            logger.error(f"Error: Could not open video source {self.camera_id}.")
            self.is_running = False
            # Create a "No Signal" frame
            no_signal = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(no_signal, "No Camera Signal", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            with lock:
                self.latest_frame = no_signal
            return

        print(f"Detector Loop Started on camera {self.camera_id}")

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Error: Could not read frame from camera.")
                # Don't exit, might be a temporary hiccup
                time.sleep(1)
                continue
                
            self.frame_count += 1
            processed_frame = frame.copy()
            
            # --- Analysis (every N frames) ---
            if self.frame_count % self.skip_frames == 0:
                try:
                    # Resize for performance and Preprocess for accuracy
                    analyze_frame = cv2.resize(frame, (640, 480))
                    analyze_frame = self._preprocess(analyze_frame)
                    
                    h_img, w_img, _ = frame.shape
                    scale_x = w_img / 640
                    scale_y = h_img / 480
                    
                    # 1. Detect & Recognize Faces
                    faces_list = self.face_rec.recognize_face(analyze_frame)
                    
                    # 2. Update Tracker with current face boxes
                    face_rects = [f['box'] for f in faces_list if f['status'] != 'no_face']
                    tracked_objects, deregistered_ids = self.tracker.update(face_rects)
                    
                    # Cleanup historical data for students who left
                    for tid in deregistered_ids:
                        if tid in self.id_map:
                            old_roll = self.id_map[tid]['roll_no']
                            if old_roll in self.students_status: 
                                self.students_status[old_roll]['status'] = 'Absent'
                                self.students_status[old_roll]['score'] = 0
                            del self.id_map[tid]
                        if tid in self.identity_history: del self.identity_history[tid]
                    
                    # 3. 1-to-1 Matching (Face to Tracker)
                    # We only process faces that can be assigned to a tracked ID
                    face_to_tid = {} # {face_index: tid}
                    used_tids = set()
                    
                    for i, face_res in enumerate(faces_list):
                        if face_res['status'] == 'no_face': continue
                        
                        x_f, y_f, w_f, h_f = face_res['box']
                        face_centroid = (int(x_f + w_f/2), int(y_f + h_f/2))
                        
                        # Find closest available tracker ID
                        best_tid = None
                        min_dist = 80 # Max pixels for a match
                        
                        for tid, centroid in tracked_objects.items():
                            if tid in used_tids: continue
                            dist_val = np.linalg.norm(np.array(face_centroid) - np.array(centroid))
                            if dist_val < min_dist:
                                min_dist = dist_val
                                best_tid = tid
                        
                        if best_tid is not None:
                            face_to_tid[i] = best_tid
                            used_tids.add(best_tid)

                    # 4. Identity Management & Analysis
                    for i, face_res in enumerate(faces_list):
                        best_tid = face_to_tid.get(i)
                        if best_tid is None: continue # Skip if no stable ID found
                        
                        # Identity Voting Logic
                        if best_tid not in self.identity_history:
                            self.identity_history[best_tid] = []
                            
                        # If recognized with high confidence, add to history
                        if face_res['status'] == 'success' and face_res.get('confidence', 0.0) >= 0.15:
                            self.identity_history[best_tid].append(face_res['roll_no'])
                        else:
                            self.identity_history[best_tid].append("Unknown")
                            
                        if len(self.identity_history[best_tid]) > self.MAX_IDENTITY_HISTORY:
                            self.identity_history[best_tid].pop(0)
                        
                        # Determine stabilized identity
                        counts = Counter(self.identity_history[best_tid])
                        voted_roll = counts.most_common(1)[0][0]
                        
                        if voted_roll != "Unknown":
                            # Pull name from recognizer map or existing id_map
                            if best_tid in self.id_map and self.id_map[best_tid]['roll_no'] == voted_roll:
                                student_info = self.id_map[best_tid]
                            else:
                                # Lookup name in FaceRecognizer's map
                                matched_data = next((v for v in self.face_rec.label_map.values() if v['roll_no'] == voted_roll), None)
                                name = matched_data['name'] if matched_data else f"Student {voted_roll}"
                                self.id_map[best_tid] = {'roll_no': voted_roll, 'name': name}
                                student_info = self.id_map[best_tid]
                        else:
                             # Default to Person_ID if consistently unknown
                             if best_tid not in self.id_map:
                                 self.id_map[best_tid] = {'roll_no': f"Person_{best_tid}", 'name': f"Student {best_tid}"}
                             student_info = self.id_map[best_tid]
                        
                        student_id = student_info['roll_no']
                        name = student_info['name']
                        
                        # --- Staggered Analysis Load Balancing ---
                        if student_id not in self.student_analysis_count:
                            self.student_analysis_count[student_id] = 0
                        
                        self.student_analysis_count[student_id] += 1
                        
                        # Only run expensive gaze/posture analysis every N skip cycles for this student
                        if self.student_analysis_count[student_id] % self.analysis_stagger == 1 or student_id not in self.students_status:
                            
                            x_a, y_a, w_a, h_a = face_res['box']
                            x, y, w, h = int(x_a * scale_x), int(y_a * scale_y), int(w_a * scale_x), int(h_a * scale_y)
                            
                            # Prepare Face Crop for Gaze Analysis (Isolated)
                            face_crop = analyze_frame[y_a:y_a+h_a, x_a:x_a+w_a]
                            if face_crop.size > 0:
                                gaze_res = self.gaze.analyze(face_crop, student_id=student_id)
                            else:
                                gaze_res = {'status': 'Unknown', 'details': 'Invalid face crop'}
                            
                            # 3. Posture Analysis
                            body_y1 = max(0, y_a - int(h_a*0.5))
                            body_y2 = min(480, y_a + int(h_a*4.0))
                            body_x1 = max(0, x_a - int(w_a*1.5))
                            body_x2 = min(640, x_a + w_a + int(w_a*1.5))
                            body_crop = analyze_frame[body_y1:body_y2, body_x1:body_x2]
                            
                            posture_res = self.posture.analyze(body_crop) if body_crop.size > 0 else {'status': 'Unknown'}
                                
                            # 4. Fusion
                            fusion_res = self.fusion.fuse(
                                face_data=face_res,
                                gaze_data=gaze_res,
                                posture_data=posture_res
                            )
                            
                            raw_state = fusion_res['state']
                            # --- Multi-frame Voting ---
                            state = self._get_voted_state(student_id, raw_state)
                            score = fusion_res['attention_score']
                            
                            # Update Last Seen
                            self.last_seen[student_id] = time.time()
                            
                            # 5. Persistent Logging & Alerts
                            is_new_event = self.logger.log_event(student_id, name, state)
                            
                            # Trigger Alert if state changed to anomalous
                            if is_new_event and state in ['Distracted', 'Sleeping', 'Bored']:
                                socketio.emit('alert_event', {
                                    'roll_no': student_id,
                                    'name': name,
                                    'message': f"{name} ({student_id}) is {state}!"
                                })
                            
                            # Update status map
                            self.students_status[student_id] = {
                                'roll_no': student_info['roll_no'], 
                                'name': name, 
                                'status': state, 
                                'score': score,
                                'accuracy': round(face_res.get('confidence', 0.0) * 100, 1) if "Person_" not in student_id else 0
                            }
                        
                        # --- Drawing (Every frame for tracked students) ---
                        current_status = self.students_status.get(student_id, {})
                        state = current_status.get('status', 'Listening')
                        x_a, y_a, w_a, h_a = face_res['box']
                        x, y, w, h = int(x_a * scale_x), int(y_a * scale_y), int(w_a * scale_x), int(h_a * scale_y)
                        
                        color = (0, 255, 0)
                        if state == 'Distracted': color = (0, 255, 255)
                        elif state == 'Sleeping': color = (0, 0, 255)
                        elif state == 'Bored': color = (255, 100, 0)
                        elif state == 'Out of Sight': color = (255, 255, 255)
                        
                        cv2.rectangle(processed_frame, (x, y), (x+w, y+h), color, 3)
                        label = f"{student_id}: {name} ({state})"
                        cv2.putText(processed_frame, label, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # 5. Handle "Out of Sight" for tracked objects without a face
                    for tid, (cX, cY) in tracked_objects.items():
                        if tid in used_tids or tid not in self.id_map: continue
                        
                        student_info = self.id_map[tid]
                        student_id = student_info['roll_no']
                        name = student_info['name']
                        
                        # If face is missing but tracker is active, person is likely hiding or turned away
                        # Log as 'Out of Sight' if not already
                        if self.students_status.get(student_id, {}).get('status') != 'Out of Sight':
                            self.students_status[student_id] = {
                                'roll_no': student_id,
                                'name': name,
                                'status': 'Out of Sight',
                                'score': 0.0,
                                'accuracy': 0
                            }
                            self.logger.log_event(student_id, name, 'Out of Sight')
                        
                        # Draw a marker at last known location
                        x, y = int(cX * scale_x), int(cY * scale_y)
                        cv2.circle(processed_frame, (x, y), 10, (255, 255, 255), -1)
                        cv2.putText(processed_frame, f"{student_id} (Out of Sight)", (x-50, y-20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                except Exception as e:
                    logger.error(f"Error in pipeline logic: {e}")

            # --- Absence Detection & Cleanup ---
            if self.frame_count % 10 == 0:
                now = time.time()
                for sid, last_time in list(self.last_seen.items()):
                    if now - last_time > self.ABSENCE_THRESHOLD:
                        # Student/Person is missing
                        current_status = self.students_status.get(sid, {})
                        if current_status and current_status.get('status') != 'Absent':
                            name = current_status.get('name', 'Student')
                            roll_label = current_status.get('roll_no', 'Unknown')
                            
                            # Log 'Distracted' (as a proxy for absent) if they were previously active
                            if current_status.get('status') != 'Distracted':
                                self.logger.log_event(sid, name, 'Distracted')
                                
                                # Send alert only for recognized students or if truly absent
                                if roll_label != "Unknown":
                                    socketio.emit('alert_event', {
                                        'roll_no': roll_label,
                                        'name': name,
                                        'message': f"{name} ({roll_label}) is absent from class!"
                                    })
                            
                            # Mark as absent or just remove from transient status list
                            if roll_label == "Unknown":
                                # For unknowns, just remove them to keep UI clean
                                if sid in self.students_status: del self.students_status[sid]
                                if sid in self.last_seen: del self.last_seen[sid]
                            else:
                                self.students_status[sid] = {
                                    'roll_no': roll_label,
                                    'name': name,
                                    'status': 'Distracted',
                                    'score': 0.0,
                                    'accuracy': 0,
                                    'details': 'Absent from class'
                                }

            # --- SocketIO ---
            if self.frame_count % (self.skip_frames * 3) == 0:
                student_list = list(self.students_status.values())
                avg_score = sum(s['score'] for s in student_list) / len(student_list) if student_list else 0
                socketio.emit('frame_data', {'students': student_list, 'avg_attention': avg_score})

            with lock:
                self.latest_frame = processed_frame
            time.sleep(0.01)

        cap.release()

# Global Singleton
detector = AttentionDetector()

def gen_frames():
    """Generator function for Flask video streaming."""
    # Ensure detector is running
    detector.start()
    
    while True:
        with lock:
            if detector.latest_frame is None:
                time.sleep(0.1)
                continue
            frame = detector.latest_frame.copy()
            
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
