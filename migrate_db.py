from app import create_app, db
from sqlalchemy import text

app = create_app()
with app.app_context():
    try:
        # Use text() to safely execute the ALTER TABLE commands
        db.session.execute(text('ALTER TABLE student ADD COLUMN department VARCHAR(64)'))
        db.session.execute(text('ALTER TABLE student ADD COLUMN year VARCHAR(10)'))
        db.session.execute(text('ALTER TABLE student ADD COLUMN image_path VARCHAR(256)'))
        db.session.commit()
        print("Database schema updated successfully.")
    except Exception as e:
        print(f"Error updating database: {e}")
        db.session.rollback()
