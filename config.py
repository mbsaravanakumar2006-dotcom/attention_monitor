import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///attention.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Attention Monitoring Thresholds
    ATTENTION_THRESHOLD_SECONDS = 3.0  # Time before flagging distraction
    SLEEP_THRESHOLD_Y = 0.6  # Normalized Y coordinate for head to be considered "down"
