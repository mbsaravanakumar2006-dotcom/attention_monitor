import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
from config import Config

db = SQLAlchemy()
socketio = SocketIO(cors_allowed_origins="*")

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)
    socketio.init_app(app)

    # Import and register blueprints/routes
    # (We will import these later to avoid circular imports during initial set up)
    from app.routes import main
    app.register_blueprint(main)
    from app import events, models
    
    # Create DB tables
    with app.app_context():
        db.create_all()

    return app
