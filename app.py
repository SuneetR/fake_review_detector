from flask import Flask, request, render_template, redirect, url_for
from fake_review_detector import predict_review, update_model  # Import the necessary functions
import os
import logging
import torch
from transformers import BertTokenizer, BertModel

app = Flask(__name__)

# Set up basic logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Define paths for storing feedback and reviews
FEEDBACK_FILE_PATH = 'feedback.txt'
REVIEWS_FILE_PATH = 'reviews.txt'

# Load BERT model and tokenizer globally to avoid reloading on each request
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Ensure BERT model is loaded on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

def store_feedback(feedback):
    """Store feedback in a file."""
    try:
        with open(FEEDBACK_FILE_PATH, 'a') as f:
            f.write(f"{feedback}\n")
        logging.info(f"Stored feedback: {feedback}")
    except Exception as e:
        logging.error(f"Failed to store feedback: {e}")

def store_review(review, review_type):
    """Store review contributions in a file."""
    try:
        with open(REVIEWS_FILE_PATH, 'a') as f:
            f.write(f"Review: {review}, Type: {review_type}\n")
        logging.info(f"Stored review: {review}, Type: {review_type}")
    except Exception as e:
        logging.error(f"Failed to store review: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    try:
        # Use BERT tokenizer and model loaded globally
        result, confidence = predict_review(review)
        logging.info(f"Prediction made: {result}, Confidence: {confidence}")
        return render_template('index.html', prediction=result, confidence=confidence)
    except Exception as e:
        logging.error(f"Failed to make prediction: {e}")
        return render_template('index.html', prediction="Error", confidence="N/A")

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback = request.form['feedback']
    store_feedback(feedback)
    logging.info(f"Received feedback: {feedback}")
    return redirect(url_for('home'))

@app.route('/submit_review', methods=['POST'])
def submit_review():
    review = request.form['review']
    review_type = request.form['review-type']
    try:
        store_review(review, review_type)
        
        # Update the model with the new review and its type
        new_label = 0 if review_type.lower() == 'genuine' else 1
        update_model(review, new_label)
        logging.info(f"Received review: {review}, Type: {review_type}, Model Updated")
    except Exception as e:
        logging.error(f"Failed to update model: {e}")
    
    return redirect(url_for('home'))

if __name__ == '__main__':
    # Ensure the files exist or create them
    try:
        if not os.path.exists(FEEDBACK_FILE_PATH):
            open(FEEDBACK_FILE_PATH, 'w').close()
        if not os.path.exists(REVIEWS_FILE_PATH):
            open(REVIEWS_FILE_PATH, 'w').close()
        logging.info("Initialized storage files.")
    except Exception as e:
        logging.error(f"Failed to initialize storage files: {e}")
    
    app.run(debug=True)
