from flask import Flask, request, jsonify
import numpy as np
from fake_review_detector import predict_fake_review  # Import the function from your script

app = Flask(__name__)

@app.route('/')
def index():
    return "Fake Review Detector API"

@app.route('/analyze', methods=['POST'])
def analyze_review():
    data = request.json  # Get the JSON data from the POST request
    review = data.get('review', '')

    if not review:
        return jsonify({'error': 'No review provided'}), 400

    try:
        # Call your prediction function from the fake_review_detector.py module
        prediction, confidence = predict_fake_review(review)

        return jsonify({'prediction': prediction, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Ensure it runs on all available IPs
