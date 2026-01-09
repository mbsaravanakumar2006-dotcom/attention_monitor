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
        # Create the recognizer - radius 1, neighbors 8 is standard, 
        # but for higher discriminative power we keep standard and tighten threshold later.
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
        
        # MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=1, # 0 for short range, 1 for long range
            min_detection_confidence=0.5
        )
        
        self.label_map = {} # {int_id: {'roll_no': ..., 'name': ...}}
        self.is_trained = False
        self.trainer_path = "app/core/trainer.xml"
        self.metadata_path = "app/core/trainer_meta.json"
        
        # CLAHE for lighting normalization
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
            
        # Initial model loading or first-time training
        if self._should_retrain():
            self.train()
        else:
            self._load_model()

    def _should_retrain(self):
        """Check if training is needed (files changed since last train)."""
        if not os.path.exists(self.trainer_path) or not os.path.exists(self.metadata_path):
            return True
        
        import json
        try:
            with open(self.metadata_path, 'r') as f:
                meta = json.load(f)
            
            # Current directory state
            image_files = [f for f in os.listdir(self.db_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if len(image_files) != meta.get('file_count', 0):
                return True
            
            # Check latest modification time
            latest_mod = max([os.path.getmtime(os.path.join(self.db_path, f)) for f in image_files]) if image_files else 0
            if latest_mod > meta.get('latest_mod', 0):
                return True
                
            return False
        except:
            return True

    def _load_model(self):
        """Load persistent model from disk."""
        import json
        try:
            self.recognizer.read(self.trainer_path)
            with open(self.metadata_path, 'r') as f:
                meta = json.load(f)
                self.label_map = {int(k): v for k, v in meta.get('label_map', {}).items()}
            self.is_trained = True
            logger.info(f"Loaded existing model from {self.trainer_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}. Retraining...")
            self.train()

    def _save_model(self, image_files):
        """Save trained model and metadata to disk."""
        import json
        try:
            self.recognizer.write(self.trainer_path)
            latest_mod = max([os.path.getmtime(os.path.join(self.db_path, f)) for f in image_files]) if image_files else 0
            meta = {
                'file_count': len(image_files),
                'latest_mod': latest_mod,
                'label_map': self.label_map
            }
            with open(self.metadata_path, 'w') as f:
                json.dump(meta, f)
            logger.info(f"Saved model to {self.trainer_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

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
            
        # 3. Brightness & Contrast adjustments
        for alpha in [0.7, 0.9, 1.1, 1.3]: # More varied brightness
            for beta in [-20, 20]: # Add contrast/offset
                adjusted = cv2.convertScaleAbs(face_img, alpha=alpha, beta=beta)
                variations.append(adjusted)
        
        # 4. Add Gaussian Noise (Simulate webcam grain)
        noise = np.random.normal(0, 5, face_img.shape).astype(np.uint8)
        variations.append(cv2.add(face_img, noise))
            
        return variations

    def get_face_crops(self, frame):
        """Extract all face crops using MediaPipe."""
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb_frame)
        
        face_crops = []
        
        if not results.detections:
            return []
            
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
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
            if crop.size == 0: continue
            
            face_crops.append((crop, [x1, y1, x2-x1, y2-y1]))
            
        return face_crops

    def train(self):
        """
        Train the recognizer with balanced samples and quality checks.
        """
        face_samples = []
        ids = []
        
        roll_to_int = {}
        current_int_id = 0
        
        image_files = [f for f in os.listdir(self.db_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if not image_files:
            logger.info("No training images found.")
            return

        # Target samples per student for balance
        SAMPLES_PER_STUDENT = 30

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
            
            # Simple Blur Detection (Laplacian Variance)
            gray_test = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray_test, cv2.CV_64F).var()
            if blur_score < 20: 
                logger.warning(f"Skipping blurry image {filename} (score: {blur_score:.1f})")
                continue

            # Detect face
            crops = self.get_face_crops(img)
            crop = crops[0][0] if crops else img # Fallback to whole image
            
            preprocessed = self.preprocess_face(crop)
            
            # Augment until we reach target
            augmented_set = self.augment_image(preprocessed)
            
            # Add images to samples
            stu_samples = 0
            while stu_samples < SAMPLES_PER_STUDENT:
                for aug_img in augmented_set:
                    if stu_samples >= SAMPLES_PER_STUDENT: break
                    face_samples.append(aug_img)
                    ids.append(label_id)
                    stu_samples += 1

        if face_samples:
            self.recognizer.train(face_samples, np.array(ids))
            self.is_trained = True
            self._save_model(image_files)
            logger.info(f"Trained on {len(face_samples)} samples for {len(self.label_map)} students.")
        else:
            logger.warning("No valid faces found for training.")

    def recognize_face(self, frame):
        """
        Recognize all faces with MediaPipe preprocessing.
        Returns a list of result dictionaries.
        """
        if not self.is_trained:
            return [{'status': 'unknown', 'message': 'Recognizer not trained'}]

        crops = self.get_face_crops(frame)
        if not crops:
            return [{'status': 'no_face', 'message': 'No face detected'}]
            
        results = []
        
        for crop, box in crops:
            preprocessed = self.preprocess_face(crop)
            label_id, confidence = self.recognizer.predict(preprocessed)
            
            # LBPH confidence: 0 is perfect, >100 is poor
            # Reduced from 85 to 65 for much higher selectivity (reduces false positives)
            if confidence < 65:
                student_data = self.label_map.get(label_id, {'roll_no': 'Unknown', 'name': 'Unknown'})
                norm_conf = max(0, (100 - confidence) / 100)
                
                results.append({
                    'status': 'success',
                    'roll_no': student_data['roll_no'],
                    'name': student_data['name'],
                    'confidence': float(norm_conf),
                    'raw_confidence': float(confidence),
                    'box': box
                })
            else:
               results.append({
                    'status': 'unknown', 
                    'roll_no': 'Unknown',
                    'name': 'Unknown',
                    'confidence': 0.0,
                    'box': box
                })
                
        return results

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
