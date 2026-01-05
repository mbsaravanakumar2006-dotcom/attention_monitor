import os
from app import create_app, socketio, db
from app.core.detector import detector

# Create the Flask application instance
app = create_app()

def initialize_database():
    """Ensure database tables are created."""
    with app.app_context():
        db.create_all()
        print("Database initialized.")

if __name__ == '__main__':
    # Initialize DB
    initialize_database()
    
    # Start the detection pipeline in its own thread
    print("Starting integrated detection pipeline...")
    detector.start(app)
    
    # Run the Flask-SocketIO server
    # Note: Use 'eventlet' or 'gevent' for production-grade concurrency
    print("Launching server on http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)
