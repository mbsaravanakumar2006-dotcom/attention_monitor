from app import create_app, db
from app.models import AttentionEvent, Student
import os

app = create_app()
with app.app_context():
    count = AttentionEvent.query.count()
    print(f"Total AttentionEvents: {count}")
    if count > 0:
        latest = AttentionEvent.query.order_by(AttentionEvent.timestamp.desc()).first()
        print(f"Latest Event: {latest.event_type} at {latest.timestamp}")
    
    student_count = Student.query.count()
    print(f"Total Students: {student_count}")
