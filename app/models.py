from datetime import datetime
from app import db

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    roll_no = db.Column(db.String(20), index=True, unique=True)
    name = db.Column(db.String(64), index=True)
    
    events = db.relationship('AttentionEvent', backref='student', lazy='dynamic')

    def __repr__(self):
        return f'<Student {self.name}>'

class AttentionEvent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'))
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    event_type = db.Column(db.String(32)) # e.g., "distracted", "sleeping", "left_seat"
    duration = db.Column(db.Float) # How long the event lasted (optional)
    
    def to_dict(self):
        return {
            'id': self.id,
            'student_id': self.student_id,
            'roll_no': self.student.roll_no if self.student else "Unknown",
            'name': self.student.name if self.student else "Unknown",
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'duration': self.duration
        }
