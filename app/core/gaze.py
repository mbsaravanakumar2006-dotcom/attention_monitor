import cv2
import mediapipe as mp
import numpy as np
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GazeTracker:
    def __init__(self):
        """
        Initialize GazeTracker with MediaPipe Face Mesh.
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        # Refine landmarks=True gives us iris landmarks
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Tracking history for temporal events like Sleeping
        # Format: {student_id: {'eyes_closed_start': timestamp, 'ear': smooth_val, 'yaw': smooth_val, 'pitch': smooth_val}}
        self.history = {}
        
        # Smoothing factor (alpha) - Lower is smoother but slower to respond
        self.alpha = 0.4
        
        # Thresholds
        self.EAR_THRESHOLD = 0.20 # Reduced from 0.22 to be more lenient
        self.SLEEP_DURATION = 3.0 # Seconds
        self.MAR_THRESHOLD = 0.12 # Threshold for "mouth open/moving"
        self.TALKING_DURATION = 5.0 # Seconds
        self.HEAD_YAW_THRESHOLD = 50 # Degrees (Reduced from 80 for more sensitivity)
        self.HEAD_PITCH_THRESHOLD = 60 # Degrees (Reduced from 70)

    def get_ear(self, landmarks, indices):
        """Calculate Eye Aspect Ratio (EAR)."""
        p1 = np.array([landmarks[indices[0]].x, landmarks[indices[0]].y])
        p4 = np.array([landmarks[indices[3]].x, landmarks[indices[3]].y])
        dist_h = np.linalg.norm(p1 - p4)
        
        p2 = np.array([landmarks[indices[1]].x, landmarks[indices[1]].y])
        p6 = np.array([landmarks[indices[5]].x, landmarks[indices[5]].y])
        dist_v1 = np.linalg.norm(p2 - p6)
        
        p3 = np.array([landmarks[indices[2]].x, landmarks[indices[2]].y])
        p5 = np.array([landmarks[indices[4]].x, landmarks[indices[4]].y])
        dist_v2 = np.linalg.norm(p3 - p5)
        
        ear = (dist_v1 + dist_v2) / (2.0 * dist_h)
        return ear

    def get_mar(self, landmarks):
        """Calculate Mouth Aspect Ratio (MAR)."""
        # Upper/Lower lip centers: 13, 14
        # Left/Right corners: 61, 291
        p13 = np.array([landmarks[13].x, landmarks[13].y])
        p14 = np.array([landmarks[14].x, landmarks[14].y])
        dist_v = np.linalg.norm(p13 - p14)
        
        p61 = np.array([landmarks[61].x, landmarks[61].y])
        p291 = np.array([landmarks[291].x, landmarks[291].y])
        dist_h = np.linalg.norm(p61 - p291)
        
        if dist_h == 0: return 0
        return dist_v / dist_h

    def get_head_pose(self, landmarks, img_w, img_h):
        """Estimate Head Pose (Yaw, Pitch, Roll)."""
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

        image_points = np.array([
            (landmarks[1].x * img_w, landmarks[1].y * img_h),
            (landmarks[152].x * img_w, landmarks[152].y * img_h),
            (landmarks[33].x * img_w, landmarks[33].y * img_h),
            (landmarks[263].x * img_w, landmarks[263].y * img_h),
            (landmarks[61].x * img_w, landmarks[61].y * img_h),
            (landmarks[291].x * img_w, landmarks[291].y * img_h)
        ], dtype="double")

        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        dist_coeffs = np.zeros((4, 1))

        try:
            success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
            rmat, jac = cv2.Rodrigues(rotation_vector)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # angles: [pitch, yaw, roll]
            return angles[0], angles[1], angles[2]
        except Exception as e:
            logger.debug(f"Pose estimation failed: {e}")
            return 0.0, 0.0, 0.0

    def analyze(self, face_image, student_id=None):
        """Analyze attention state for a single face crop."""
        try:
            h, w, c = face_image.shape
            rgb_frame = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return {'status': 'Unknown', 'details': 'No face detected'}
                
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = face_landmarks.landmark
            
            # 1. EAR Calculation
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            
            raw_ear = (self.get_ear(landmarks, left_eye_indices) + self.get_ear(landmarks, right_eye_indices)) / 2.0
            
            # 2. Head Pose Calibration
            raw_pitch, raw_yaw, raw_roll = self.get_head_pose(landmarks, w, h)
            
            # --- Temporal Smoothing (EMA) ---
            if student_id:
                if student_id not in self.history:
                    self.history[student_id] = {
                        'eyes_closed_start': None,
                        'talking_start': None,
                        'ear': raw_ear,
                        'yaw': raw_yaw,
                        'pitch': raw_pitch,
                        'mar': self.get_mar(landmarks)
                    }
                
                h_data = self.history[student_id]
                h_data['ear'] = self.alpha * raw_ear + (1 - self.alpha) * h_data['ear']
                h_data['yaw'] = self.alpha * raw_yaw + (1 - self.alpha) * h_data['yaw']
                h_data['pitch'] = self.alpha * raw_pitch + (1 - self.alpha) * h_data['pitch']
                
                raw_mar = self.get_mar(landmarks)
                h_data['mar'] = self.alpha * raw_mar + (1 - self.alpha) * h_data.get('mar', raw_mar)
                
                ear = h_data['ear']
                yaw = h_data['yaw']
                pitch = h_data['pitch']
                mar = h_data['mar']
            else:
                ear, yaw, pitch = raw_ear, raw_yaw, raw_pitch

            # Determine State
            eyes_closed = ear < self.EAR_THRESHOLD
            state = "Listening"
            details = "Looking at screen/teacher"
            
            current_time = time.time()
            if student_id:
                if eyes_closed:
                    if self.history[student_id]['eyes_closed_start'] is None:
                        self.history[student_id]['eyes_closed_start'] = current_time
                    elif current_time - self.history[student_id]['eyes_closed_start'] > self.SLEEP_DURATION:
                        state = "Sleeping"
                        details = "Eyes closed > 3s"
                else:
                    self.history[student_id]['eyes_closed_start'] = None
                    
            if state != "Sleeping":
                if eyes_closed:
                    state = "Drowsy/Blinking" 
                elif abs(yaw) > self.HEAD_YAW_THRESHOLD:
                    state = "Distracted"
                    details = "Looking away (Yaw)"
                elif abs(pitch) > self.HEAD_PITCH_THRESHOLD:
                    state = "Distracted"
                    details = "Looking up/down (Pitch)"
                
                # Check for Talking
                if student_id:
                    if mar > self.MAR_THRESHOLD:
                        if self.history[student_id].get('talking_start') is None:
                            self.history[student_id]['talking_start'] = current_time
                        elif current_time - self.history[student_id]['talking_start'] > self.TALKING_DURATION:
                            state = "Talking"
                            details = "Talking/Mouth movement > 5s"
                    else:
                        self.history[student_id]['talking_start'] = None
                
            return {
                'status': state,
                'details': details,
                'metrics': {
                    'ear': ear,
                    'yaw': yaw,
                    'pitch': pitch,
                    'mar': mar
                }
            }
            
        except Exception as e:
            logger.error(f"Error in gaze analysis: {e}")
            return {'status': 'Error', 'details': str(e)}

if __name__ == "__main__":
    # Test script
    tracker = GazeTracker()
    cap = cv2.VideoCapture(0)
    
    # Fake ID for testing temporal features
    test_id = "user_1"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        result = tracker.analyze(frame, student_id=test_id)
        
        # Display
        color = (0, 255, 0)
        if result['status'] == 'Distracted': color = (0, 165, 255)
        elif result['status'] == 'Sleeping': color = (0, 0, 255)
        
        cv2.putText(frame, f"State: {result['status']}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Msg: {result.get('details','')}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if 'metrics' in result:
            m = result['metrics']
            stats = f"EAR: {m['ear']:.2f} Y: {m['yaw']:.0f} P: {m['pitch']:.0f}"
            cv2.putText(frame, stats, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow("Attention Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
