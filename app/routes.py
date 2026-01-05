from flask import Blueprint, render_template, jsonify, Response
from app import db
from app.models import AttentionEvent
from app.core.detector import gen_frames

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('dashboard.html')

@main.route('/api/stats')
def stats():
    # Return recent events
    events = AttentionEvent.query.order_by(AttentionEvent.timestamp.desc()).limit(50).all()
    return jsonify([e.to_dict() for e in events])

@main.route('/report')
def report():
    return render_template('report.html')

@main.route('/api/attention_history')
def attention_history():
    # Fetch data for the graph - last 30 minutes of average attention
    # This is a simplified version; in a real app, you might aggregate by minute
    events = AttentionEvent.query.order_by(AttentionEvent.timestamp.desc()).limit(100).all()
    # Mocking aggregated history for now or just returning recent events to be processed by JS
    return jsonify([e.to_dict() for e in events])

@main.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
