import sqlite3
import os

db_path = 'instance/attention.db'

def view():
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}. Please run 'python app.py' first to initialize it.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("="*40)
    print("   ATTENTION MONITOR DATABASE VIEW")
    print("="*40)
    
    print("\n--- Registered Students ---")
    try:
        cursor.execute("SELECT id, roll_no, name FROM student")
        students = cursor.fetchall()
        if not students:
            print("No students registered yet.")
        for s in students:
            print(f"ID: {s[0]} | Roll: {s[1]} | Name: {s[2]}")
    except Exception as e:
        print(f"Error reading students: {e}")
        
    print("\n--- Recent Attention Events (Last 20) ---")
    try:
        # Join with student table to get names
        query = """
        SELECT s.name, s.roll_no, e.event_type, e.timestamp 
        FROM attention_event e
        JOIN student s ON e.student_id = s.id
        ORDER BY e.timestamp DESC LIMIT 20
        """
        cursor.execute(query)
        events = cursor.fetchall()
        if not events:
            print("No events logged yet.")
        for e in events:
            print(f"[{e[3]}] {e[0]} ({e[1]}) -> {e[2]}")
    except Exception as e:
        print(f"Error reading events: {e}")
        
    print("\n" + "="*40)
    conn.close()

if __name__ == "__main__":
    view()
