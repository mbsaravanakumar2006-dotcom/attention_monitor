from app import create_app, db
from app.models import Student
import os

app = create_app()
with app.app_context():
    # Specifically target student 075 to rename to Saravana if it's currently showing filename
    s = Student.query.filter_by(roll_no='075').first()
    if s and 'WIN_' in s.name:
        print(f"Fixing 075 name: {s.name} -> Saravana")
        s.name = "Saravana"
        
        # Rename file if it exists
        if s.image_path:
            old_full_path = os.path.join(app.root_path, 'static', s.image_path)
            if os.path.exists(old_full_path):
                ext = os.path.splitext(s.image_path)[1]
                new_filename = f"075_Saravana{ext}"
                new_rel_path = f"faces/{new_filename}"
                new_full_path = os.path.join(app.root_path, 'static', 'faces', new_filename)
                
                os.rename(old_full_path, new_full_path)
                s.image_path = new_rel_path
    
    db.session.commit()
    
    # Final retrain with the new names
    from app.core.detector import detector
    detector.start(app)
    stu_map = {st.roll_no: st.name for st in Student.query.all()}
    detector.face_rec.train(roll_to_name_map=stu_map)
    print("Final retrain complete.")
