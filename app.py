from flask import Flask, request, render_template, redirect, url_for, jsonify
from fake_review_detector import predict_review, update_model
import os

app = Flask(__name__)

# Paths for storing feedback and reviews
FEEDBACK_FILE_PATH = 'feedback.txt'
REVIEWS_FILE_PATH = 'reviews.txt'

def store_feedback(feedback):
    """Store feedback in a file."""
    with open(FEEDBACK_FILE_PATH, 'a') as f:
        f.write(f"{feedback}\n")

def store_review(review, review_type):
    """Store review contributions in a file."""
    with open(REVIEWS_FILE_PATH, 'a') as f:
        f.write(f"Review: {review}, Type: {review_type}\n")

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests for the review analysis."""
    data = request.get_json()
    review = data.get('review', '')
    
    # Check if review text is provided
    if not review:
        return jsonify({'error': 'Review text is missing'}), 400
    
    # Get prediction and confidence from the model
    prediction, confidence = predict_review(review)
    return jsonify({'prediction': prediction, 'confidence': confidence})

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    """Handle feedback submission."""
    feedback = request.form['feedback']
    store_feedback(feedback)
    print(f"Received feedback: {feedback}")
    return jsonify({'status': 'success', 'message': 'Feedback received'})

@app.route('/submit_review', methods=['POST'])
def submit_review():
    """Handle review submission for model update."""
    review = request.form['review']
    review_type = request.form['review-type']
    store_review(review, review_type)
    
    # Update the model with the new review and its label
    new_label = 1 if review_type.strip().lower() == 'true' else 0
    update_model(review, new_label)
    print(f"Received review: {review}, Type: {review_type}, Model Updated")
    
    return jsonify({'status': 'success', 'message': 'Review received and model updated'})

if __name__ == '__main__':
    # Ensure feedback and reviews files exist or create them
    if not os.path.exists(FEEDBACK_FILE_PATH):
        open(FEEDBACK_FILE_PATH, 'w').close()
    if not os.path.exists(REVIEWS_FILE_PATH):
        open(REVIEWS_FILE_PATH, 'w').close()
    
    app.run(debug=True)
