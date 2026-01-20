from app import create_app, db
from app.models import Student
import os

app = create_app()
with app.app_context():
    students = Student.query.all()
    for s in students:
        if s.image_path and s.image_path.startswith('uploads/students/'):
            old_filename = os.path.basename(s.image_path)
            # The naming convention in FaceRecognizer is roll_no_name.jpg
            # Let's try to fix existing records to point to the correct folder
            new_filename = f"{s.roll_no}_{s.name}{os.path.splitext(old_filename)[1]}"
            s.image_path = f"faces/{new_filename}"
            print(f"Updated {s.roll_no} path to {s.image_path}")
    
    db.session.commit()
    print("Database paths updated.")
