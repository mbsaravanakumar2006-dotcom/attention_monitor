import logging
from datetime import datetime
from app import db, create_app
from app.models import Student, AttentionEvent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EventLogger:
    def __init__(self, app_context=None):
        """
        Initialize EventLogger.
        
        Args:
            app_context: Flask application context (optional).
                         If not provided, tries to create one (useful for independent scripts).
        """
        self.state_cache = {} # {student_name: {'state': current_state, 'start_time': timestamp}}
        self.app = app_context
        
        if not self.app:
            try:
                # Attempt to get current app or create one
                from flask import current_app
                if current_app:
                    self.app = current_app
            except:
                pass
                
        # If still no app, we might need to create one for DB access in standalone mode
        # Example: self.app = create_app()

    def log_event(self, roll_no, name, new_state):
        """
        Log event only if state changes.
        Returns True if a new event was logged.
        """
        if not roll_no:
            return False

        current_time = datetime.utcnow()
        
        # Check cache to avoid spamming the DB
        if roll_no in self.state_cache:
            last_state = self.state_cache[roll_no]['state']
            if last_state == new_state:
                return False 
        
        logger.info(f"State Change: {roll_no} ({name}) -> {new_state}")
        self.state_cache[roll_no] = {
            'state': new_state,
            'start_time': current_time
        }
        
        try:
            # Check for current app context (works in request threads)
            from flask import current_app
            if current_app:
                self._save_to_db(roll_no, name, new_state, current_time)
            # Check for stored app instance (works in background threads)
            elif self.app:
                with self.app.app_context():
                    self._save_to_db(roll_no, name, new_state, current_time)
            # Try to get app from external source if possible, or just fail
            else:
                logger.warning(f"No app context available to log event for {roll_no}. Data not saved.")
            
            return True
        except Exception as e:
            logger.error(f"Failed to log event to DB: {e}")
            return False

    def _save_to_db(self, roll_no, name, state, timestamp):
        try:
            # Find or Create Student by Roll No
            student = Student.query.filter_by(roll_no=roll_no).first()
            if not student:
                student = Student(roll_no=roll_no, name=name)
                db.session.add(student)
                db.session.commit()
            elif student.name != name:
                # Update name if it changed (optional)
                student.name = name
                db.session.commit()
            
            # Create Event
            event = AttentionEvent(
                student_id=student.id,
                event_type=state,
                timestamp=timestamp,
                duration=0.0
            )
            
            db.session.add(event)
            db.session.commit()
            logger.info(f"Logged event {event.id} for {roll_no}")
            
        except Exception as e:
            db.session.rollback()
            raise e

if __name__ == "__main__":
    # Test script - assumes app factory pattern or reachable app
    # We need to setup a fake app context for this test to work with the real DB
    # or mock it. For now, we will perform a dry run.
    
    print("Initializing Event Logger...")
    # NOTE: This will fail if 'app' package cannot be imported correctly relative to here
    # Run with python -m app.core.event_logger
    
    logger = EventLogger()
    
    print("Logging 'Listening' for Alice...")
    logger.log_event("Alice", "Listening")
    
    print("Logging 'Listening' for Alice (Duplicate)...")
    logger.log_event("Alice", "Listening") # Should not log
    
    print("Logging 'Distracted' for Alice...")
    logger.log_event("Alice", "Distracted") # Should log
