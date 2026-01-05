import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BehaviorFusion:
    def __init__(self):
        """
        Initialize Behavior Fusion Engine.
        """
        # Confidence Weights
        self.weights = {'eye_open': 0.5, 'head_pose': 0.3, 'posture': 0.2}
        self.MIN_CONFIDENCE = 0.6 # Ignore data below this
        
    def fuse(self, face_data, gaze_data, posture_data, motion_level=0.5):
        """Fuse modalities with confidence thresholding."""
        
        # 1. Extract Inputs
        face_conf = face_data.get('confidence', 1.0) # Recognition confidence
        gaze_status = gaze_data.get('status', 'Unknown')
        gaze_metrics = gaze_data.get('metrics', {})
        ear = gaze_metrics.get('ear', 0.3)
        
        posture_status = posture_data.get('status', 'Unknown')
        posture_conf = posture_data.get('confidence', 1.0)
        
        # 2. State Logic with Confidence Filtering
        final_state = "Attentive"
        reason = "Engaged"
        
        # Priority 1: Sleeping (High Confidence from Gaze Required)
        if gaze_status == 'Sleeping':
             final_state = "Sleeping"
             reason = "Eyes closed > 3s"
             
        # Priority 2: Distracted (Head / Gaze)
        elif gaze_status == 'Distracted':
             final_state = "Distracted"
             reason = gaze_data.get('details', 'Looking away')
             
        # Priority 3: Posture Override (Only if confidence is high)
        elif posture_conf > self.MIN_CONFIDENCE:
            if posture_status == 'Head Down':
                final_state = "Sleeping" if ear < 0.23 else "Distracted"
                reason = "Head down"
            elif posture_status == 'Leaning':
                final_state = "Distracted"
                reason = "Slumping/Leaning"
            elif posture_status == 'Slouching' and motion_level < 0.2:
                final_state = "Bored"
                reason = "Inactive & Slouching"

        # 3. Dynamic Attention Score (Normalized 0-100)
        # Weighting: Eye (60%), Head Pose (20%), Posture (20%)
        score_eye = 1.0 if not (ear < 0.23) else 0.3
        score_head = 1.0 if gaze_status == 'Attentive' else 0.6 if gaze_status == 'Drowsy/Blinking' else 0.4
        score_posture = 1.0 if posture_status == 'Good' else 0.7 if posture_status == 'Leaning' else 0.5
        
        raw_score = (score_eye * 0.6) + (score_head * 0.2) + (score_posture * 0.2)
        
        return {
            'state': final_state,
            'reason': reason,
            'confidence': round(face_conf * 0.5 + posture_conf * 0.5, 2),
            'attention_score': round(raw_score * 100, 1)
        }

if __name__ == "__main__":
    # Test Script
    fusion = BehaviorFusion()
    
    # Test Case 1: Perfect Student
    print("Test 1: Listening")
    res1 = fusion.fuse(
        face_data={'confidence': 0.95},
        gaze_data={'status': 'Attentive', 'metrics': {'ear': 0.3}},
        posture_data={'status': 'Good', 'confidence': 0.9},
        motion_level=0.5
    )
    print(res1)
    
    # Test Case 2: Sleeping
    print("\nTest 2: Sleeping")
    res2 = fusion.fuse(
        face_data={'confidence': 0.9},
        gaze_data={'status': 'Sleeping'}, # Triggered by timer in gaze module
        posture_data={'status': 'Head Down', 'confidence': 0.8},
        motion_level=0.0
    )
    print(res2)
    
    # Test Case 3: Distracted
    print("\nTest 3: Distracted")
    res3 = fusion.fuse(
        face_data={'confidence': 0.9},
        gaze_data={'status': 'Distracted', 'details': 'Looking Left'},
        posture_data={'status': 'Good', 'confidence': 0.9},
        motion_level=0.6
    )
    print(res3)
