import logging
import sys
from app import create_app, db
# Ensure we are using a test config or existing one
# We will just use the default create_app logic

def test_clear_data_route():
    print("Initializing App for Testing...")
    app = create_app()
    app.config['TESTING'] = True
    
    with app.test_client() as client:
        print("Sending POST request to /api/clear_data...")
        response = client.post('/api/clear_data')
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Data: {response.data}")
        
        if response.status_code == 200:
            print("SUCCESS: Route is registered and working.")
        elif response.status_code == 404:
            print("FAILURE: Route not found (404). Check registration.")
        else:
            print(f"FAILURE: Unexpected status {response.status_code}")

if __name__ == "__main__":
    try:
        test_clear_data_route()
    except Exception as e:
        print(f"CRITICAL EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
