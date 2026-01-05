import os
import cv2
import numpy as np
import logging
import mediapipe as mp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRecognizer:
    def __init__(self, db_path="app/static/faces"):
        """
        Initialize the FaceRecognizer using OpenCV LBPH and MediaPipe Face Detection.
        """
        self.db_path = db_path
        # Create the recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=1, # 0 for short range, 1 for long range
            min_detection_confidence=0.5
        )
        
        self.label_map = {} # {int_id: {'roll_no': ..., 'name': ...}}
        self.is_trained = False
        
        # CLAHE for lighting normalization
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
            
        self.train()

    def preprocess_face(self, face_img):
        """Standardize face image: Grayscale -> CLAHE -> Resize."""
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
            
        # Apply CLAHE
        standardized = self.clahe.apply(gray)
        # Resize to fixed resolution
        standardized = cv2.resize(standardized, (128, 128))
        return standardized

    def augment_image(self, face_img):
        """Generate synthetic variations of the face image."""
        variations = [face_img]
        
        # 1. Horizontal Flip
        variations.append(cv2.flip(face_img, 1))
        
        # 2. Rotations (+/- 10 degrees)
        h, w = face_img.shape[:2]
        center = (w // 2, h // 2)
        for angle in [-10, 10]:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(face_img, M, (w, h))
            variations.append(rotated)
            
        # 3. Brightness adjustments
        for alpha in [0.8, 1.2]:
            adjusted = cv2.convertScaleAbs(face_img, alpha=alpha, beta=0)
            variations.append(adjusted)
            
        return variations

    def get_face_crop(self, frame):
        """Extract primary face crop using MediaPipe."""
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb_frame)
        
        if not results.detections:
            return None, None
            
        # Take the detection with largest area (primary person)
        best_det = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height)
        
        bbox = best_det.location_data.relative_bounding_box
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)
        
        # Padding
        pad_x, pad_y = int(bw * 0.1), int(bh * 0.1)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: return None, None
        
        return crop, [x1, y1, x2-x1, y2-y1]

    def train(self):
        """
        Train the recognizer with MediaPipe detection and Data Augmentation.
        """
        face_samples = []
        ids = []
        
        roll_to_int = {}
        current_int_id = 0
        
        image_files = [f for f in os.listdir(self.db_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if not image_files:
            logger.info("No training images found.")
            return

        for filename in image_files:
            basename = filename.split('.')[0]
            if '_' in basename:
                roll_no, name = basename.split('_', 1)
            else:
                roll_no = "UNK"
                name = basename
                
            if roll_no not in roll_to_int:
                roll_to_int[roll_no] = current_int_id
                self.label_map[current_int_id] = {'roll_no': roll_no, 'name': name}
                current_int_id += 1
            
            label_id = roll_to_int[roll_no]
            img_path = os.path.join(self.db_path, filename)
            
            img = cv2.imread(img_path)
            if img is None: continue
            
            # Detect face in the training image
            crop, _ = self.get_face_crop(img)
            if crop is None:
                # Fallback: if MediaPipe fails on specific static image, use whole image (assuming it's a tight crop)
                crop = img
            
            preprocessed = self.preprocess_face(crop)
            
            # Augment
            augmented_set = self.augment_image(preprocessed)
            for aug_img in augmented_set:
                face_samples.append(aug_img)
                ids.append(label_id)

        if face_samples:
            self.recognizer.train(face_samples, np.array(ids))
            self.is_trained = True
            logger.info(f"Trained on {len(face_samples)} samples (original + aug) for {len(self.label_map)} students.")
        else:
            logger.warning("No valid faces found for training.")

    def recognize_face(self, frame):
        """
        Recognize face with MediaPipe preprocessing.
        """
        if not self.is_trained:
            return {'status': 'unknown', 'message': 'Recognizer not trained'}

        crop, box = self.get_face_crop(frame)
        if crop is None:
            return {'status': 'no_face', 'message': 'No face detected'}
            
        preprocessed = self.preprocess_face(crop)
        label_id, confidence = self.recognizer.predict(preprocessed)
        
        # LBPH confidence: 0 is perfect, >100 is poor
        # We'll use a threshold of 85 for "trained well" consistency
        if confidence < 85:
            student_data = self.label_map.get(label_id, {'roll_no': 'Unknown', 'name': 'Unknown'})
            norm_conf = max(0, (100 - confidence) / 100)
            
            return {
                'status': 'success',
                'roll_no': student_data['roll_no'],
                'name': student_data['name'],
                'confidence': float(norm_conf),
                'raw_confidence': float(confidence),
                'box': box
            }
        else:
            return {
                'status': 'unknown', 
                'roll_no': 'Unknown',
                'name': 'Unknown',
                'confidence': 0.0,
                'box': box
            }

    def register_student(self, frame, student_id):
        """
        Register a new student.
        """
        file_path = os.path.join(self.db_path, f"{student_id}.jpg")
        cv2.imwrite(file_path, frame)
        logger.info(f"Saved image for {student_id}. Re-training...")
        self.train()
        return True

if __name__ == "__main__":
    # Internal Test
    recon = FaceRecognizer()
    print("Self-Test Finished.")
