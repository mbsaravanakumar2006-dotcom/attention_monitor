from flask import Blueprint, render_template, jsonify, Response
from app import db
from app.models import AttentionEvent
from app.core.detector import gen_frames

from sqlalchemy.orm import joinedload

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('dashboard.html')

@main.route('/api/stats')
def stats():
    # Return recent events - use joinedload to prevent N+1 queries
    events = AttentionEvent.query.options(joinedload(AttentionEvent.student))\
        .order_by(AttentionEvent.timestamp.desc()).limit(100).all()
    return jsonify([e.to_dict() for e in events])

@main.route('/report')
def report():
    return render_template('report.html')

@main.route('/api/attention_summary')
def attention_summary():
    # Aggregated attention data for the chart (past 24h)
    from datetime import datetime, timedelta
    from sqlalchemy import func
    
    # Define scoring logic (consistent with frontend report.js)
    # Focused/Listening: 100, Distracted: 30, Sleeping: 0, Bored: 40, Default: 50
    score_case = db.case(
        (AttentionEvent.event_type.in_(['Focused', 'Attentive', 'Listening']), 100),
        (AttentionEvent.event_type == 'Distracted', 30),
        (AttentionEvent.event_type == 'Sleeping', 0),
        (AttentionEvent.event_type == 'Bored', 40),
        else_=50
    )
    
    # Use localized time for query if possible, but stick to UTC for consistency with logger
    # SQLite logic: strftime('%H:%M', timestamp)
    summary = db.session.query(
        func.strftime('%H:%M', AttentionEvent.timestamp).label('time_str'),
        func.avg(score_case).label('avg_score')
    ).filter(AttentionEvent.timestamp > datetime.utcnow() - timedelta(hours=24))\
     .group_by('time_str')\
     .order_by('time_str').all()
    
    if not summary:
        return jsonify([])
        
    return jsonify([{'time': s.time_str, 'score': round(s.avg_score, 1)} for s in summary])

@main.route('/api/export_report')
def export_report():
    import csv
    import io
    from flask import make_response
    
    events = AttentionEvent.query.options(joinedload(AttentionEvent.student))\
        .order_by(AttentionEvent.timestamp.desc()).all()
    
    proxy = io.StringIO()
    writer = csv.writer(proxy)
    writer.writerow(['Timestamp', 'Roll No', 'Student Name', 'Event Type'])
    
    for e in events:
        writer.writerow([
            e.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            e.student.roll_no if e.student else "Unknown",
            e.student.name if e.student else "Unknown",
            e.event_type
        ])
    
    output = make_response(proxy.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=attention_report.csv"
    output.headers["Content-type"] = "text/csv"
    return output

@main.route('/api/attention_history')
def attention_history():
    # Fetch data for the table - limit to 500 recent events for performance
    events = AttentionEvent.query.options(joinedload(AttentionEvent.student))\
        .order_by(AttentionEvent.timestamp.desc()).limit(500).all()
    return jsonify([e.to_dict() for e in events])

@main.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
