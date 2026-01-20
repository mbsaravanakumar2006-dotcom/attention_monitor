from app import create_app, db
from app.models import Student
import os

app = create_app()
with app.app_context():
    students = Student.query.all()
    faces_dir = os.path.join(app.root_path, 'static', 'faces')
    
    for s in students:
        if not s.image_path:
            continue
            
        current_rel_path = s.image_path
        current_full_path = os.path.join(app.root_path, 'static', current_rel_path)
        
        if os.path.exists(current_full_path):
            ext = os.path.splitext(current_rel_path)[1]
            # Construct standard filename: RollNo_Name.ext
            new_filename = f"{s.roll_no}_{s.name}{ext}".replace(' ', '_') # Replace spaces just in case
            new_rel_path = f"faces/{new_filename}"
            new_full_path = os.path.join(faces_dir, new_filename)
            
            if current_full_path != new_full_path:
                print(f"Renaming {current_rel_path} -> {new_rel_path}")
                if os.path.exists(new_full_path):
                    os.remove(new_full_path) # Avoid collision if already exists
                os.rename(current_full_path, new_full_path)
                s.image_path = new_rel_path
        else:
            print(f"File not found: {current_full_path}")

    db.session.commit()
    print("Database and physical files updated.")

    # Trigger retraining
    from app.core.detector import detector
    detector.start(app)
    detector.face_rec.train()
    print("Face recognition model retrained.")
