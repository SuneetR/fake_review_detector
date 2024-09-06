from flask import Flask, request, render_template, redirect, url_for
from fake_review_detector import predict_review, update_model  # Import prediction and model update functions
import os
import nltk

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

# Define paths for storing feedback and reviews
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
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get review and product type from the form
    review = request.form.get('review')
    product_type = request.form.get('product_type')

    # Ensure review is provided
    if not review:
        return render_template('index.html', error="Please provide a review.")

    # Call the prediction function
    result, confidence = predict_review(review, product_type)
    
    # Render template with prediction results
    return render_template('index.html', prediction=result, confidence=confidence)


@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback = request.form.get('feedback')

    if feedback:
        store_feedback(feedback)
        print(f"Received feedback: {feedback}")
    else:
        print("No feedback provided.")

    return redirect(url_for('home'))


@app.route('/submit_review', methods=['POST'])
def submit_review():
    review = request.form.get('review')
    review_type = request.form.get('review_type')

    # Validate that both review and review_type are provided
    if not review or not review_type:
        return render_template('index.html', error="Please provide both review and review type.")

    # Store the review and its type (fake or genuine)
    store_review(review, review_type)

    # Update the model with the new review and its type
    # Assuming 'fake' = 1 and 'genuine' = 0
    new_label = 1 if review_type.lower() == 'fake' else 0
    update_model(review, new_label)
    
    print(f"Received review: {review}, Type: {review_type}, Model Updated")

    return redirect(url_for('home'))


if __name__ == '__main__':
    # Ensure the files exist or create them
    if not os.path.exists(FEEDBACK_FILE_PATH):
        open(FEEDBACK_FILE_PATH, 'w').close()
    if not os.path.exists(REVIEWS_FILE_PATH):
        open(REVIEWS_FILE_PATH, 'w').close()

    app.run(debug=True)
