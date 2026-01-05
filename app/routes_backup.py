from flask import render_template, jsonify, Response
from app import app, db
from app.models import AttentionEvent

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/stats')
def stats():
    # Return recent events
    events = AttentionEvent.query.order_by(AttentionEvent.timestamp.desc()).limit(50).all()
    return jsonify([e.to_dict() for e in events])

# Placeholder for video feed if we decide to stream processed frames via HTTP
# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
