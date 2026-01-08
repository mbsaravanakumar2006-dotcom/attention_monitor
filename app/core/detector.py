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
        
        # Modules
        # Ensure models are loaded
        print("Loading Face Recognizer...")
        self.face_rec = FaceRecognizer()
        print("Loading Posture Analyzer...")
        self.posture = PostureAnalyzer()
        print("Loading Gaze Tracker...")
        self.gaze = GazeTracker()
        print("Loading Fusion Engine...")
        self.fusion = BehaviorFusion()
        print("Loading Event Logger...")
        self.logger = EventLogger()
        
        # --- Accuracy & Stability Enhancements ---
        self.state_history = {} # {id: [state1, state2, ...]} (buffer size 15)
        self.VOTE_WINDOW = 15
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        # Runtime Data
        self.students_status = {} # {id: {status: '...', name: '...', score: 90}}
        self.last_seen = {} # {id: timestamp}
        self.ABSENCE_THRESHOLD = 7.0 # Seconds
        self.skip_frames = 2 # Process every Nth frame
        self.frame_count = 0

    def start(self, app=None):
        if not self.is_running:
            self.is_running = True
            if app:
                self.logger.app = app
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
                    face_res = self.face_rec.recognize_face(analyze_frame)
                    
                    if face_res['status'] == 'no_face':
                        continue

                    if face_res['status'] == 'success':
                        # Lenient Confidence Thresholding
                        # face_res['confidence'] is (100-conf)/100.
                        # 0.15 means raw confidence was 85.
                        if face_res.get('confidence', 0.0) < 0.15:
                            roll_no = "Unknown"
                            name = "Unrecognized Student"
                        else:
                            roll_no = face_res['roll_no']
                            name = face_res['name']
                    else:
                        # Even if recognition is 'unknown', we still want to track the face
                        roll_no = "Unknown"
                        name = "New Student"
                        
                    student_id = roll_no # Using roll_no as unique identifier for tracking
                    
                    x_a, y_a, w_a, h_a = face_res['box']
                    x, y, w, h = int(x_a * scale_x), int(y_a * scale_y), int(w_a * scale_x), int(h_a * scale_y)
                    
                    # 2. Gaze Analysis
                    gaze_res = self.gaze.analyze(analyze_frame, student_id=student_id)
                    
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
                    self.last_seen[roll_no] = time.time()
                    
                    # 5. Persistent Logging & Alerts
                    is_new_event = self.logger.log_event(roll_no, name, state)
                    
                    # Trigger Alert if state changed to anomalous
                    if is_new_event and state in ['Distracted', 'Sleeping', 'Bored']:
                        socketio.emit('alert_event', {
                            'roll_no': roll_no,
                            'name': name,
                            'message': f"{name} ({roll_no}) is {state}!"
                        })
                    
                    # 6. UI Updates
                    color = (0, 255, 0)
                    if state == 'Distracted': color = (0, 255, 255)
                    elif state == 'Sleeping': color = (0, 0, 255)
                    elif state == 'Bored': color = (255, 100, 0)
                    
                    cv2.rectangle(processed_frame, (x, y), (x+w, y+h), color, 3)
                    label = f"{roll_no}: {name} ({state})"
                    cv2.putText(processed_frame, label, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    self.students_status[roll_no] = {
                        'roll_no': roll_no, 
                        'name': name, 
                        'status': state, 
                        'score': score,
                        'accuracy': round(face_res.get('confidence', 0.0) * 100, 1) if roll_no != "Unknown" else 0
                    }

                except Exception as e:
                    logger.error(f"Error in pipeline logic: {e}")

            # --- Absence Detection ---
            # Check every 10 frames to see if any previously active students are now missing
            if self.frame_count % 10 == 0:
                now = time.time()
                for roll_no, last_time in list(self.last_seen.items()):
                    if now - last_time > self.ABSENCE_THRESHOLD:
                        # Student is missing
                        current_status = self.students_status.get(roll_no, {})
                        if current_status.get('status') != 'Distracted' and current_status.get('status') != 'Absent':
                            name = current_status.get('name', 'Student')
                            
                            # Update status
                            self.students_status[roll_no] = {
                                'roll_no': roll_no,
                                'name': name,
                                'status': 'Distracted',
                                'score': 0.0,
                                'accuracy': 0,
                                'details': 'Absent from class'
                            }
                            
                            # Log to DB
                            self.logger.log_event(roll_no, name, 'Distracted')
                            
                            # Alert
                            socketio.emit('alert_event', {
                                'roll_no': roll_no,
                                'name': name,
                                'message': f"{name} ({roll_no}) is absent from class!"
                            })

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
