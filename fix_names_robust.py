from app import create_app, db
from app.models import Student
import os
import glob

app = create_app()
with app.app_context():
    students = Student.query.all()
    faces_dir = os.path.join(app.root_path, 'static', 'faces')
    
    for s in students:
        # 1. Try to find the file mentioned in DB
        current_full_path = os.path.join(app.root_path, 'static', s.image_path if s.image_path else "")
        
        # 2. If not found, try to find ANY file in faces/ starting with roll_no_
        if not s.image_path or not os.path.exists(current_full_path):
            pattern = os.path.join(faces_dir, f"{s.roll_no}_*")
            matches = glob.glob(pattern)
            if matches:
                current_full_path = matches[0]
                print(f"Found loose match for {s.roll_no}: {current_full_path}")
            else:
                print(f"No file found for student {s.roll_no}")
                continue
        
        # 3. Rename to standard format: faces/RollNo_Name.ext
        ext = os.path.splitext(current_full_path)[1]
        new_filename = f"{s.roll_no}_{s.name}{ext}".replace(' ', '_')
        new_rel_path = f"faces/{new_filename}"
        new_full_path = os.path.join(faces_dir, new_filename)
        
        if current_full_path != new_full_path:
            print(f"Standardizing: {os.path.basename(current_full_path)} -> {new_filename}")
            if os.path.exists(new_full_path):
                os.remove(new_full_path)
            os.rename(current_full_path, new_full_path)
            s.image_path = new_rel_path

    db.session.commit()
    print("Alignment complete.")

    # Retrain
    from app.core.detector import detector
    detector.start(app)
    detector.face_rec.train()
    print("Retraining triggered.")
