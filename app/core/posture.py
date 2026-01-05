import cv2
import mediapipe as mp
import numpy as np
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PostureAnalyzer:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize the PostureAnalyzer with MediaPipe Pose.
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            static_image_mode=False,
            model_complexity=1
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def calculate_angle(self, a, b, c):
        """
        Calculate angle between three points.
        a, b, c are [x, y] coordinates.
        b is the vertex.
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle

    def analyze(self, frame):
        """
        Analyze posture in the given frame (should be a crop of a single person).
        
        Args:
            frame (numpy.ndarray): Image frame.
            
        Returns:
            dict: Analysis results containing 'status', 'confidence', 'details'.
        """
        try:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            
            if not results.pose_landmarks:
                return {'status': 'Unknown', 'confidence': 0.0, 'details': 'No pose detected'}
            
            landmarks = results.pose_landmarks.landmark
            
            # Extract key landmarks
            # MediaPipe Pose Landmarks: https://google.github.io/mediapipe/solutions/pose.html
            nose = [landmarks[self.mp_pose.PoseLandmark.NOSE.value].x,
                    landmarks[self.mp_pose.PoseLandmark.NOSE.value].y]
            
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            
            left_ear = [landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y]
            right_ear = [landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].y]
                         
            # --- Analysis Logic ---
            
            state = "Good"
            reasons = []
            
            # 1. Head Down Detection (Vector from shoulders midpoint to nose)
            shoulder_midpoint = [(left_shoulder[0] + right_shoulder[0]) / 2,
                                 (left_shoulder[1] + right_shoulder[1]) / 2]
            
            # Check vertical alignment of nose relative to shoulders
            # If nose is too low (large Y value) relative to shoulders, head is down
            # Normally nose Y < shoulder Y (Y increases downwards)
            
            # Simple heuristic: Nose Y should be significantly above shoulder Y
            # Threshold depends on distance, so normalize by shoulder width
            shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
            if shoulder_width == 0: shoulder_width = 0.001
            nose_shoulder_dist_y = shoulder_midpoint[1] - nose[1] # Positive if nose is above
            
            # If nose is close to shoulder line vertically
            if nose_shoulder_dist_y < 0.05 * shoulder_width:
                 state = "Head Down"
                 reasons.append("Head lowered")
            
            # 2. Slouching / Leaning (Shoulder Tilt)
            # Calculate angle of shoulder line relative to horizontal
            # Using basic trig: tan(theta) = dy / dx
            dy = left_shoulder[1] - right_shoulder[1]
            dx = left_shoulder[0] - right_shoulder[0]
            if dx != 0:
                shoulder_slope = dy/dx
                shoulder_angle = math.degrees(math.atan(shoulder_slope))
                if abs(shoulder_angle) > 15: # 15 degrees tilt
                    state = "Leaning" if state == "Good" else state
                    reasons.append(f"Shoulders tilted {abs(shoulder_angle):.1f} deg")
            
            # 3. Slouching (Forward Hunch) - Hard from 2D front view
            # Proxy: Distance between ears and shoulders becomes small?
            # Or use visibility of neck? MediaPipe doesn't give neck explicitly.
            # We can check vertical distance between ears and shoulders.
            ear_y_avg = (left_ear[1] + right_ear[1]) / 2
            neck_length_y = shoulder_midpoint[1] - ear_y_avg
            
            # If neck seems too short, maybe hunching? This is very calibration sensitive.
            # We'll stick to robust metrics for now.
            
            # Confidence based on visibility of key landmarks
            avg_vis = (landmarks[self.mp_pose.PoseLandmark.NOSE.value].visibility +
                       landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility +
                       landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility) / 3
            
            return {
                'status': state,
                'confidence': avg_vis,
                'details': ", ".join(reasons) if reasons else "Upright posture",
                'landmarks': {
                    'nose': nose,
                    'shoulders': [left_shoulder, right_shoulder]
                }
            }
            
        except Exception as e:
            logger.error(f"Error in posture analysis: {e}")
            return {'status': 'Error', 'confidence': 0.0, 'details': str(e)}

if __name__ == "__main__":
    # Test script
    analyzer = PostureAnalyzer()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        result = analyzer.analyze(frame)
        
        # Visualize
        text = f"{result['status']} ({result['confidence']:.2f})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        if 'details' in result:
             cv2.putText(frame, result['details'], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # Draw simple landmarks if available
        if 'landmarks' in result and result.get('landmarks'):
             h, w, _ = frame.shape
             nose = result['landmarks']['nose']
             shoulders = result['landmarks']['shoulders']
             
             cv2.circle(frame, (int(nose[0]*w), int(nose[1]*h)), 5, (0, 0, 255), -1)
             cv2.line(frame, 
                      (int(shoulders[0][0]*w), int(shoulders[0][1]*h)), 
                      (int(shoulders[1][0]*w), int(shoulders[1][1]*h)), 
                      (255, 0, 0), 2)

        cv2.imshow("Posture Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
