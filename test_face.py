from app.core.face_recognition import FaceRecognizer
import os

def test_training():
    print("Testing Face Recognition Training Overhaul...")
    recon = FaceRecognizer()
    
    # Check if trained
    if recon.is_trained:
        print("Model trained successfully.")
        print(f"Students in map: {recon.label_map}")
    else:
        print("Model training failed.")

if __name__ == "__main__":
    test_training()
