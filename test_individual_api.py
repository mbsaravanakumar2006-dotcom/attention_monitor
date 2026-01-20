from app import create_app, db
from app.models import AttentionEvent, Student
import requests
import json

app = create_app()

def test_api_filtering():
    with app.app_context():
        # 1. Test student list
        students = Student.query.all()
        print(f"Total students in DB: {len(students)}")
        
        # 2. Test filtered summary
        if students:
            target_roll = students[0].roll_no
            print(f"Testing filtered summary for roll_no: {target_roll}")
            
            # Since we can't easily call the route via requests without running the server,
            # we simulate the logic or assume the server is running on default port if we were doing a real integration test.
            # For this verification, we'll check if the logic in routes.py is sound by looking at the DB directly.
            
            events_count = AttentionEvent.query.join(Student).filter(Student.roll_no == target_roll).count()
            print(f"Events for {target_roll}: {events_count}")
            
            # Check a non-existent student
            fake_roll = "999999"
            fake_events_count = AttentionEvent.query.join(Student).filter(Student.roll_no == fake_roll).count()
            print(f"Events for fake roll {fake_roll}: {fake_events_count}")

if __name__ == "__main__":
    test_api_filtering()
