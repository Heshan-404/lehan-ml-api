import joblib
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Load the model
try:
    model = joblib.load("model.joblib")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")


@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()

        # Expecting 'features' key in JSON with a list of 20 numerical values
        if 'features' not in data:
            return jsonify({"error": "No 'features' key in request JSON"}), 400

        features = data['features']

        if len(features) != 20:
            return jsonify({
                "error": f"Model expects 20 features, but got {len(features)}"
            }), 400

        # Reshape to a 2D array for prediction: (1, 20)
        features_array = np.array(features).reshape(1, -1)

        # Predict using the loaded pipeline
        prediction = model.predict(features_array)

        return jsonify({"prediction": float(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})


if __name__ == '__main__':
    # Run the app on 0.0.0.0 to be accessible outside the container
    app.run(host='0.0.0.0', port=8080)
