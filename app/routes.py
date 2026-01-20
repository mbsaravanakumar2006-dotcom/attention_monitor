import os
from flask import Blueprint, render_template, jsonify, Response, request, redirect, url_for, session, flash, current_app
from app import db
from app.models import AttentionEvent, User, Student
from app.core.detector import gen_frames
from functools import wraps
from werkzeug.utils import secure_filename

from sqlalchemy.orm import joinedload

main = Blueprint('main', __name__)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('main.login'))
        return f(*args, **kwargs)
    return decorated_function

@main.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('main.dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Successfully logged in!', 'success')
            return redirect(url_for('main.dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
            
    return render_template('login.html')

@main.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('Successfully logged out.', 'info')
    return redirect(url_for('main.index'))

@main.route('/')
def index():
    return render_template('home.html')

@main.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@main.route('/api/stats')
def stats():
    # Return recent events - use joinedload to prevent N+1 queries
    events = AttentionEvent.query.options(joinedload(AttentionEvent.student))\
        .order_by(AttentionEvent.timestamp.desc()).limit(100).all()
    return jsonify([e.to_dict() for e in events])

@main.route('/report')
@login_required
def report():
    return render_template('report.html')

@main.route('/api/attention_summary')
def attention_summary():
    # Aggregated attention data for the chart (past 24h)
    from datetime import datetime, timedelta
    from sqlalchemy import func
    
    roll_no = request.args.get('roll_no')
    
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
    
    query = db.session.query(
        time_group.label('time_str'),
        func.avg(score_case).label('avg_score')
    ).filter(AttentionEvent.timestamp > datetime.now() - timedelta(hours=24))
    
    if roll_no:
        query = query.join(Student).filter(Student.roll_no == roll_no)
        
    summary = query.group_by('time_str')\
     .order_by('time_str').all()
    
    return jsonify([{'time': s.time_str, 'score': round(s.avg_score, 1)} for s in summary])

@main.route('/api/students_list')
def students_list():
    # Only return students who have data or are registered
    students = Student.query.all()
    return jsonify([{'roll_no': s.roll_no, 'name': s.name} for s in students])

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

@main.route('/students')
@login_required
def students():
    students_list = Student.query.all()
    return render_template('manage_students.html', students=students_list)

@main.route('/add_student', methods=['GET', 'POST'])
@login_required
def add_student():
    if request.method == 'POST':
        roll_no = request.form.get('roll_no')
        name = request.form.get('name')
        department = request.form.get('department')
        year = request.form.get('year')
        file = request.files.get('image')

        if not roll_no or not name or not department or not year or not file:
            flash('All fields are required!', 'danger')
            return redirect(url_for('main.add_student'))

        # Check if student already exists
        if Student.query.filter_by(roll_no=roll_no).first():
            flash('Student with this Roll No already exists!', 'danger')
            return redirect(url_for('main.add_student'))

        # Save image to 'faces' folder with correct naming convention for detector
        ext = os.path.splitext(file.filename)[1]
        filename = secure_filename(f"{roll_no}_{name}{ext}")
        upload_folder = os.path.join(current_app.root_path, 'static', 'faces')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)
        
        # Save to DB
        new_student = Student(
            roll_no=roll_no,
            name=name,
            department=department,
            year=year,
            image_path=f"faces/{filename}"
        )
        db.session.add(new_student)
        db.session.commit()
        
        # Trigger face recognition retraining
        from app.core.detector import detector
        detector.start(current_app._get_current_object())
        stu_map = {s.roll_no: s.name for s in Student.query.all()}
        detector.face_rec.train(roll_to_name_map=stu_map)
        
        flash('Student added successfully and models updated!', 'success')
        return redirect(url_for('main.students'))

    return render_template('add_student.html')

@main.route('/edit_student/<int:id>', methods=['GET', 'POST'])
@login_required
def edit_student(id):
    student = Student.query.get_or_404(id)
    if request.method == 'POST':
        old_roll = student.roll_no
        old_name = student.name
        
        student.roll_no = request.form.get('roll_no')
        student.name = request.form.get('name')
        student.department = request.form.get('department')
        student.year = request.form.get('year')
        
        file = request.files.get('image')
        # Handle image update or renaming of existing file if roll_no/name changed
        if (file and file.filename != '') or (student.roll_no != old_roll or student.name != old_name):
            upload_folder = os.path.join(current_app.root_path, 'static', 'faces')
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            # If user uploaded a NEW file
            if file and file.filename != '':
                # Delete old image if it exists
                if student.image_path:
                    old_path = os.path.join(current_app.root_path, 'static', student.image_path)
                    if os.path.exists(old_path):
                        os.remove(old_path)
                
                ext = os.path.splitext(file.filename)[1]
                filename = secure_filename(f"{student.roll_no}_{student.name}{ext}")
                file_path = os.path.join(upload_folder, filename)
                file.save(file_path)
                student.image_path = f"faces/{filename}"
            
            # If ONLY roll_no or name changed, rename the existing file
            elif student.image_path:
                current_full_path = os.path.join(current_app.root_path, 'static', student.image_path)
                if os.path.exists(current_full_path):
                    ext = os.path.splitext(student.image_path)[1]
                    new_filename = secure_filename(f"{student.roll_no}_{student.name}{ext}")
                    new_full_path = os.path.join(upload_folder, new_filename)
                    os.rename(current_full_path, new_full_path)
                    student.image_path = f"faces/{new_filename}"

        try:
            db.session.commit()
            
            # Trigger face recognition retraining
            from app.core.detector import detector
            detector.start(current_app._get_current_object())
            stu_map = {s.roll_no: s.name for s in Student.query.all()}
            detector.face_rec.train(roll_to_name_map=stu_map)
            
            flash('Student updated and models retrained!', 'success')
            return redirect(url_for('main.students'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating student: {str(e)}', 'danger')

    return render_template('edit_student.html', student=student)

@main.route('/delete_student/<int:id>', methods=['POST'])
@login_required
def delete_student(id):
    student = Student.query.get_or_404(id)
    try:
        # Delete image file
        if student.image_path:
            full_path = os.path.join(current_app.root_path, 'static', student.image_path)
            if os.path.exists(full_path):
                os.remove(full_path)
        
        db.session.delete(student)
        db.session.commit()
        flash('Student deleted successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting student: {str(e)}', 'danger')
    
    return redirect(url_for('main.students'))
