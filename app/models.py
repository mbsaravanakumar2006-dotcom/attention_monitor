from datetime import datetime
from app import db
from werkzeug.security import generate_password_hash, check_password_hash

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    roll_no = db.Column(db.String(20), index=True, unique=True)
    name = db.Column(db.String(64), index=True)
    department = db.Column(db.String(64))
    year = db.Column(db.String(10))
    image_path = db.Column(db.String(256))
    
    events = db.relationship('AttentionEvent', backref='student', lazy='dynamic', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Student {self.name}>'

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    password_hash = db.Column(db.String(128))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

class AttentionEvent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), index=True)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.now)
    event_type = db.Column(db.String(32), index=True) # e.g., "distracted", "sleeping", "left_seat"
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
