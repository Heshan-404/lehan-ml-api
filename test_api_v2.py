import requests
import json

# Replace with your railway URL after you start it
# URL = "https://lehan-ml-api-production.up.railway.app"
URL = "http://127.0.0.1:8080" 

def test_predict():
    endpoint = f"{URL}/predict"
    
    # Send 20 features for the machine learning engine
    payload = {
        "features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    }
    
    print(f"Testing Unified Analytics Engine: {endpoint}")
    try:
        response = requests.post(endpoint, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Full Unified Response: {json.dumps(response.json(), indent=4)}")
    except Exception as e:
        print(f"Error calling /predict: {e}")

def test_health():
    endpoint = f"{URL}/health"
    print(f"Testing System Health: {endpoint}")
    try:
        response = requests.get(endpoint)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=4)}")
    except Exception as e:
        print(f"Error calling /health: {e}")

if __name__ == "__main__":
    print("\n--- Running AuraCart MLOps Verification ---")
    test_health()
    print("\n--- Testing Unified Mandate (Tasks 3.2, 3.3, 3.5) ---")
    test_predict()
    print("\nVerification Complete.")
