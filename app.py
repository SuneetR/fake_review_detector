from flask import Flask, request, render_template, jsonify, redirect, url_for
from fake_review_detector import predict_review  # Import your model's prediction function
import os
import nltk
import logging
import gc
from flask_cors import CORS, cross_origin

# Ensure only necessary NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from your front-end

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths for storing feedback and reviews
FEEDBACK_FILE_PATH = 'feedback.txt'
REVIEWS_FILE_PATH = 'reviews.txt'

def store_feedback(feedback):
    try:
        with open(FEEDBACK_FILE_PATH, 'a') as f:
            f.write(f"{feedback}\n")
        logger.info(f"Feedback stored: {feedback}")
    except Exception as e:
        logger.error(f"Error storing feedback: {e}")

def store_review(review, review_type):
    try:
        with open(REVIEWS_FILE_PATH, 'a') as f:
            f.write(f"Review: {review}, Type: {review_type}\n")
        logger.info(f"Review stored: {review}, Type: {review_type}")
    except Exception as e:
        logger.error(f"Error storing review: {e}")

@app.route('/')
def home():
    return render_template('index.html')

# Route for predicting review using AJAX
@app.route('/analyze', methods=['POST'])
@cross_origin()  # Enable CORS for this specific route
def analyze_review():
    try:
        data = request.json
        review = data.get('review')

        if not review:
            return jsonify({'error': 'Review content is missing.'}), 400

        # Get prediction from the model
        prediction, confidence = predict_review(review)

        return jsonify({
            'prediction': prediction,
            'confidence': confidence
        })

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback = request.form.get('feedback')

    if feedback:
        store_feedback(feedback)
        logger.info(f"Received feedback: {feedback}")
    else:
        logger.info("No feedback provided.")

    gc.collect()
    return redirect(url_for('home'))

@app.route('/submit_review', methods=['POST'])
def submit_review():
    review = request.form.get('review')
    review_type = request.form.get('review_type')

    if not review or not review_type:
        return render_template('index.html', error="Please provide both review and review type.")

    try:
        store_review(review, review_type)
        logger.info(f"Received review: {review}, Type: {review_type}")
        del review, review_type
        gc.collect()
        return redirect(url_for('home'))

    except Exception as e:
        logger.error(f"Error while storing the review: {e}")
        gc.collect()
        return render_template('index.html', error=f"An error occurred while storing the review: {str(e)}")

if __name__ == '__main__':
    # Create the feedback and reviews files if they don't exist
    if not os.path.exists(FEEDBACK_FILE_PATH):
        open(FEEDBACK_FILE_PATH, 'w').close()
    if not os.path.exists(REVIEWS_FILE_PATH):
        open(REVIEWS_FILE_PATH, 'w').close()

    app.run(debug=False)
