from flask import Flask, request, render_template, redirect, url_for
from fake_review_detector import predict_review, update_model  # Import prediction and model update functions
import os
import nltk
import logging

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths for storing feedback and reviews
FEEDBACK_FILE_PATH = 'feedback.txt'
REVIEWS_FILE_PATH = 'reviews.txt'

def store_feedback(feedback):
    """Store feedback in a file."""
    try:
        with open(FEEDBACK_FILE_PATH, 'a') as f:
            f.write(f"{feedback}\n")
        logger.info(f"Feedback stored: {feedback}")
    except Exception as e:
        logger.error(f"Error storing feedback: {e}")

def store_review(review, review_type):
    """Store review contributions in a file."""
    try:
        with open(REVIEWS_FILE_PATH, 'a') as f:
            f.write(f"Review: {review}, Type: {review_type}\n")
        logger.info(f"Review stored: {review}, Type: {review_type}")
    except Exception as e:
        logger.error(f"Error storing review: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get review and product type from the form
    review = request.form.get('review')
    product_type = request.form.get('product_type')

    # Ensure review is provided
    if not review:
        return render_template('index.html', error="Please provide a review.")

    try:
        # Call the prediction function from fake_review_detector
        result, confidence = predict_review(review, product_type)

        # Render template with prediction results
        return render_template('index.html', prediction=result, confidence=f"{confidence:.2f}")

    except Exception as e:
        # Handle any errors that occur during prediction
        logger.error(f"Error during prediction: {e}")
        return render_template('index.html', error=f"An error occurred during prediction: {str(e)}")

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback = request.form.get('feedback')

    if feedback:
        store_feedback(feedback)
        logger.info(f"Received feedback: {feedback}")
    else:
        logger.info("No feedback provided.")

    return redirect(url_for('home'))

@app.route('/submit_review', methods=['POST'])
def submit_review():
    review = request.form.get('review')
    review_type = request.form.get('review_type')

    # Validate that both review and review_type are provided
    if not review or not review_type:
        return render_template('index.html', error="Please provide both review and review type.")

    try:
        # Store the review and its type (fake or genuine)
        store_review(review, review_type)

        # Update the model with the new review and its type
        # Assuming 'fake' = 1 and 'genuine' = 0
        new_label = 1 if review_type.lower() == 'fake' else 0
        update_model(review, new_label)

        logger.info(f"Received review: {review}, Type: {review_type}, Model Updated")
        return redirect(url_for('home'))

    except Exception as e:
        # Handle any errors that occur during model update
        logger.error(f"Error while updating the model: {e}")
        return render_template('index.html', error=f"An error occurred while updating the model: {str(e)}")

if __name__ == '__main__':
    # Ensure the feedback and reviews files exist or create them
    if not os.path.exists(FEEDBACK_FILE_PATH):
        open(FEEDBACK_FILE_PATH, 'w').close()
    if not os.path.exists(REVIEWS_FILE_PATH):
        open(REVIEWS_FILE_PATH, 'w').close()

    app.run(debug=True)
