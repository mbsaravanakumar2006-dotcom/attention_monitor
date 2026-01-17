from app import create_app, db
from app.models import User
import sys

def create_admin(username, password):
    app = create_app()
    with app.app_context():
        # Check if user already exists
        user = User.query.filter_by(username=username).first()
        if user:
            print(f"User {username} already exists. Updating password...")
            user.set_password(password)
        else:
            print(f"Creating new admin user: {username}")
            user = User(username=username)
            user.set_password(password)
            db.session.add(user)
        
        try:
            db.session.commit()
            print("Admin user created successfully!")
        except Exception as e:
            db.session.rollback()
            print(f"Error: {e}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python create_admin.py <username> <password>")
        sys.argv = ['create_admin.py', 'admin', 'admin123'] # Default for convenience if user just runs it
        print("Using default: admin / admin123")
    
    create_admin(sys.argv[1], sys.argv[2])
