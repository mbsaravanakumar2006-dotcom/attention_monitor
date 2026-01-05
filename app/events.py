from app import socketio

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

# We will emit events from the core detector loop, 
# so we might not need many incoming events from client yet.
