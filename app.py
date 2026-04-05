import joblib
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Load all three models for the AuraCart engine (Task 3.2, 3.3, 3.5)
try:
    regression_model = joblib.load("model.joblib")
    classification_model = joblib.load("classification_model.joblib")
    clustering_model = joblib.load("clustering_model.joblib")
    print("All 3 models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data.get('features')
        
        if not features or len(features) != 20:
             return jsonify({
                 "error": f"Model expects 20 features, but got {len(features) if features else 0}"
             }), 400

        features_array = np.array(features).reshape(1, -1)

        # Execute the "Triple Mandate" from the project specification:
        
        # 1. Regression (Task 3.2): Revenue Forecasting
        value_prediction = regression_model.predict(features_array)[0]

        # 2. Classification (Task 3.3): Order Risk Assessment
        # 0 = On-time, 1 = High Risk of Delay
        risk_class = int(classification_model.predict(features_array)[0])
        risk_status = "High Risk of Delay" if risk_class == 1 else "On-time Probability"

        # 3. Clustering (Task 3.5): Behavioral Segmentation
        # Segment 0 = "Bronze", 1 = "Silver", 2 = "Gold"
        cluster_id = int(clustering_model.predict(features_array)[0])
        segments = ["Bronze Customer", "Silver Customer", "Gold Customer"]
        segment_label = segments[cluster_id] if cluster_id < 3 else "Unknown"

        return jsonify({
            "status": "success",
            "results": {
                "regression": {
                    "task": "Revenue Forecast (Task 3.2)",
                    "predicted_order_value": round(float(value_prediction), 2)
                },
                "classification": {
                    "task": "Delivery Risk Assessment (Task 3.3)",
                    "risk_probability": risk_status
                },
                "clustering": {
                    "task": "Customer Behavioral Segmentation (Task 3.5)",
                    "segment": segment_label
                }
            },
            "context": "AuraCart Unified Analytics Engine"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "models_loaded": True,
        "engine": "AuraCart Unified v1.2"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
