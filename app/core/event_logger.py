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
        self.state_cache = {} # {roll_no: {'state': current_state, 'start_time': timestamp}}
        self.pending_states = {} # {roll_no: {'state': new_state, 'first_seen': timestamp}}
        self.STABILITY_THRESHOLD = 3.0 # Seconds a state must be stable before logging
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
        Log event only if state changes and remains stable for STABILITY_THRESHOLD.
        Returns True if a new event was PERMANENTLY logged to DB.
        """
        if not roll_no:
            return False

        current_time = datetime.now()
        
        # 1. Check if state is already the active (logged) state
        if roll_no in self.state_cache:
            if self.state_cache[roll_no]['state'] == new_state:
                # State hasn't changed from what we last officially logged
                # Clear any pending state for this student as they've returned to "normal"
                if roll_no in self.pending_states:
                    del self.pending_states[roll_no]
                return False 

        # 2. Check stability for new state
        if roll_no not in self.pending_states or self.pending_states[roll_no]['state'] != new_state:
            # First time seeing this new state, or state changed again before stabilizing
            self.pending_states[roll_no] = {
                'state': new_state,
                'first_seen': current_time
            }
            # Special Case: For real-time SocketIO alerts, we might want to return True here 
            # or handle alerts separately. The caller (detector.py) handles alerts.
            # We return False because it's not yet *logged* to the DB.
            return False

        # 3. Check if threshold reached
        elapsed = (current_time - self.pending_states[roll_no]['first_seen']).total_seconds()
        if elapsed < self.STABILITY_THRESHOLD:
            return False
            
        # 4. Threshold reached! Official state change.
        logger.info(f"State Stabilized: {roll_no} ({name}) -> {new_state} (after {elapsed:.1f}s)")
        self.state_cache[roll_no] = {
            'state': new_state,
            'start_time': self.pending_states[roll_no]['first_seen']
        }
        del self.pending_states[roll_no] # Clear pending
        
        try:
            # Check for current app context (works in request threads)
            from flask import current_app
            if current_app:
                self._save_to_db(roll_no, name, new_state, current_time)
            # Check for stored app instance (works in background threads)
            elif self.app:
                with self.app.app_context():
                    self._save_to_db(roll_no, name, new_state, current_time)
            else:
                logger.warning(f"No app context available to log event for {roll_no}. Data not saved.")
            
            return True
        except Exception as e:
            logger.error(f"Failed to log event to DB: {e}")
            return False

    def _save_to_db(self, roll_no, name, state, timestamp):
        try:
            # 1. Update previous event's duration if it exists
            self._update_previous_event_duration(roll_no, timestamp)

            # 2. Find or Create Student
            student = Student.query.filter_by(roll_no=roll_no).first()
            if not student:
                student = Student(roll_no=roll_no, name=name)
                db.session.add(student)
                db.session.commit()
            elif student.name != name:
                student.name = name
                db.session.commit()
            
            # 3. Create New Event
            event = AttentionEvent(
                student_id=student.id,
                event_type=state,
                timestamp=timestamp,
                duration=0.0 # Initial duration
            )
            
            db.session.add(event)
            db.session.commit()
            logger.info(f"Logged new event {event.id} for {roll_no} ({state})")
            
        except Exception as e:
            db.session.rollback()
            raise e

    def _update_previous_event_duration(self, roll_no, current_time):
        """Update the duration of the most recent event for this student."""
        try:
            # Find the most recent event for this student
            last_event = db.session.query(AttentionEvent).join(Student)\
                .filter(Student.roll_no == roll_no)\
                .order_by(AttentionEvent.timestamp.desc()).first()

            if last_event and last_event.duration == 0.0:
                duration = (current_time - last_event.timestamp).total_seconds()
                # Minimum duration of 1s to avoid 0.0 in DB if clock is slightly off
                last_event.duration = max(1.0, round(duration, 1))
                db.session.commit()
                logger.info(f"Updated duration for event {last_event.id}: {last_event.duration}s")
        except Exception as e:
            logger.error(f"Failed to update previous event duration: {e}")
            db.session.rollback()

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
