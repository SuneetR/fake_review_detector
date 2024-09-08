from flask import Flask, request, jsonify
import numpy as np
import joblib  # Assuming you are using joblib to load your model

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model.pkl')  # Update with your actual model path

@app.route('/analyze', methods=['POST'])
def analyze_review():
    data = request.json  # Get the JSON data from the POST request
    review = data.get('review', '')

    if not review:
        return jsonify({'error': 'No review provided'}), 400

    try:
        # Preprocess the review before making a prediction
        # Assuming you have some preprocessing steps
        # Convert input to a suitable format (e.g., NumPy array)
        input_data = np.array([review])  # Wrap review in a list and convert to a NumPy array

        # Get the prediction and confidence score
        prediction = model.predict(input_data)  # Ensure model input is a NumPy array
        confidence = model.predict_proba(input_data).max()  # Confidence score

        # Convert the NumPy results to Python lists if needed
        prediction_list = prediction.tolist() if isinstance(prediction, np.ndarray) else prediction
        confidence_list = confidence.tolist() if isinstance(confidence, np.ndarray) else confidence

        return jsonify({'prediction': prediction_list[0], 'confidence': confidence_list})  # Access the first element
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Ensure it runs on all available IPs
