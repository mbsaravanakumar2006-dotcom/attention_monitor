from app import create_app, db
from app.models import AttentionEvent
from datetime import datetime, timedelta
from sqlalchemy import func
import json

app = create_app()
with app.app_context():
    score_case = db.case(
        (AttentionEvent.event_type.in_(['Focused', 'Attentive', 'Listening']), 100),
        (AttentionEvent.event_type == 'Distracted', 30),
        (AttentionEvent.event_type == 'Sleeping', 0),
        (AttentionEvent.event_type == 'Bored', 40),
        else_=50
    )
    
    summary = db.session.query(
        func.strftime('%H:%M', AttentionEvent.timestamp).label('time_str'),
        func.avg(score_case).label('avg_score')
    ).filter(AttentionEvent.timestamp > datetime.utcnow() - timedelta(hours=24))\
     .group_by('time_str')\
     .order_by('time_str').all()
    
    result = [{'time': s.time_str, 'score': round(s.avg_score, 1)} for s in summary]
    print(f"Summary result count: {len(result)}")
    print(json.dumps(result[:5], indent=2))

    # Also check history
    events = AttentionEvent.query.limit(5).all()
    print(f"History sample: {[e.to_dict() for e in events]}")
