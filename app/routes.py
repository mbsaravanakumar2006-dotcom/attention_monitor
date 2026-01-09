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
    
    # 10-minute interval logic for SQLite
    # This reduces 1440 points to ~144 points for faster processing and faster chart rendering
    time_group = func.strftime('%H:', AttentionEvent.timestamp) + \
                 (func.cast(func.strftime('%M', AttentionEvent.timestamp), db.Integer) / 10).cast(db.String) + '0'

    score_case = db.case(
        (AttentionEvent.event_type.in_(['Focused', 'Attentive', 'Listening']), 100),
        (AttentionEvent.event_type == 'Distracted', 30),
        (AttentionEvent.event_type == 'Sleeping', 0),
        (AttentionEvent.event_type == 'Bored', 40),
        else_=50
    )
    
    summary = db.session.query(
        time_group.label('time_str'),
        func.avg(score_case).label('avg_score')
    ).filter(AttentionEvent.timestamp > datetime.now() - timedelta(hours=24))\
     .group_by('time_str')\
     .order_by('time_str').all()
    
    return jsonify([{'time': s.time_str, 'score': round(s.avg_score, 1)} for s in summary])

@main.route('/api/export_report')
def export_report():
    import csv
    from io import StringIO
    from flask import stream_with_context
    
    # Use streaming to avoid memory issues with large logs
    def generate():
        data = StringIO()
        writer = csv.writer(data)
        writer.writerow(['Timestamp', 'Roll No', 'Student Name', 'Event Type'])
        yield data.getvalue()
        data.seek(0)
        data.truncate(0)

        # Batch process 500 rows at a time
        offset = 0
        batch_size = 500
        while True:
            events = AttentionEvent.query.options(joinedload(AttentionEvent.student))\
                .order_by(AttentionEvent.timestamp.desc())\
                .offset(offset).limit(batch_size).all()
            
            if not events:
                break
                
            for e in events:
                writer.writerow([
                    e.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    e.student.roll_no if e.student else "Unknown",
                    e.student.name if e.student else "Unknown",
                    e.event_type
                ])
                yield data.getvalue()
                data.seek(0)
                data.truncate(0)
            
            offset += batch_size

    response = Response(stream_with_context(generate()), mimetype='text/csv')
    response.headers["Content-Disposition"] = "attachment; filename=attention_report.csv"
    return response

@main.route('/api/attention_history')
def attention_history():
    # Only return top 200 for lightning fast initial load
    events = AttentionEvent.query.options(joinedload(AttentionEvent.student))\
        .order_by(AttentionEvent.timestamp.desc()).limit(150).all()
    return jsonify([e.to_dict() for e in events])

@main.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@main.route('/api/clear_data', methods=['POST'])
def clear_data():
    try:
        num_rows_deleted = db.session.query(AttentionEvent).delete()
        db.session.commit()
        return jsonify({'status': 'success', 'message': f'Deleted {num_rows_deleted} records.'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500
