import joblib
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# AuraCart Deployment: All 4 models for the unified engine
try:
    regression_model = joblib.load("model.joblib")
    delivery_status_model = joblib.load("delivery_status_model.joblib")
    customer_segment_model = joblib.load("customer_segment_model.joblib")
    behavioral_clustering_model = joblib.load("behavioral_clustering_model.joblib")
    print("All 4 models for the AuraCart Mandate have been loaded successfully.")
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

        # 1. Regression (Task 3.2): Monetary Value Prediction
        value_pred = regression_model.predict(features_array)[0]

        # 2. Classification (Task 3.3a): Delivery Status prediction
        # Target classes: Delivered, Shipped, Pending, Returned
        delivery_class = int(delivery_status_model.predict(features_array)[0])
        delivery_status = ["Delivered", "Shipped", "Pending", "Returned"][delivery_class]

        # 3. Classification (Task 3.3b): Customer Segment prediction (Supervised)
        # Target classes: New, Returning, VIP
        segment_class = int(customer_segment_model.predict(features_array)[0])
        customer_segment = ["New Customer", "Returning Customer", "VIP Customer"][segment_class]

        # 4. Clustering (Task 3.5): Behavioral Cluster (Unsupervised)
        # Behavioral categorization based on patterns
        cluster_id = int(behavioral_clustering_model.predict(features_array)[0])
        segments = ["Bronze behavior", "Silver behavior", "Gold behavior"]
        behavior_label = segments[cluster_id] if cluster_id < 3 else "Unknown"

        return jsonify({
            "status": "success",
            "results": {
                "regression": {
                    "task": "Revenue Forecast (Task 3.2)",
                    "predicted_order_value": round(float(value_pred), 2)
                },
                "delivery_classification": {
                    "task": "Order Logistic Risk (Task 3.3a)",
                    "status": delivery_status
                },
                "segmentation_classification": {
                    "task": "Strategic Marketing Target (Task 3.3b)",
                    "segment": customer_segment
                },
                "behavioral_clustering": {
                    "task": "Latent Behavioral Grouping (Task 3.5)",
                    "behavior": behavior_label
                }
            },
            "context": "AuraCart Unified Unified Production Engine v1.5"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "models_count": 4,
        "engine": "AuraCart MLOps Unified"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
