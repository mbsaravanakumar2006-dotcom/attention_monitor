from app import create_app, socketio, db
from app.core.detector import detector

app = create_app()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    detector.start()
    socketio.run(app, debug=True, use_reloader=False)
