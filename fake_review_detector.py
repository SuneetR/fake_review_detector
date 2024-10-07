import os
import logging
import gc
import string
import nltk
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pickle
from flask import Flask, request, jsonify

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Ensure required NLTK data is downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Initialize Flask app
app = Flask(__name__)

# Model path for saving/loading
MODEL_PATH = 'trained_model.pkl'

# Tokenizer
def basic_tokenize(text):
    tokens = text.lower().split()
    tokens = [word.strip(string.punctuation) for word in tokens]
    return tokens

# Preprocessing function
def preprocess_text(text):
    tokens = basic_tokenize(text)
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(cleaned_tokens)

# Function to load reviews from CSV
def load_reviews(file_path):
    logging.debug(f"Loading reviews from {file_path}")
    return pd.read_csv(file_path)

# Model setup
tfidf_vectorizer = TfidfVectorizer(max_features=2000)
pca = PCA(n_components=50)
stacking_model = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ],
    final_estimator=LogisticRegression()
)

# Function to train the model
def train_model(reviews, labels):
    try:
        logging.debug("Starting model training...")
        cleaned_reviews = reviews.apply(preprocess_text)
        tfidf_features = tfidf_vectorizer.fit_transform(cleaned_reviews)
        tfidf_features_pca = pca.fit_transform(tfidf_features.toarray())

        stacking_model.fit(tfidf_features_pca, labels)
        logging.debug("Model training complete.")

        # Save the trained model to disk
        with open(MODEL_PATH, 'wb') as model_file:
            pickle.dump((stacking_model, tfidf_vectorizer, pca), model_file)
        gc.collect()
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

# Function to load the pre-trained model
def load_trained_model():
    try:
        if os.path.exists(MODEL_PATH):
            logging.debug(f"Loading trained model from {MODEL_PATH}")
            with open(MODEL_PATH, 'rb') as model_file:
                global stacking_model, tfidf_vectorizer, pca
                stacking_model, tfidf_vectorizer, pca = pickle.load(model_file)
            logging.debug("Model loaded successfully.")
        else:
            logging.error(f"Model file {MODEL_PATH} not found. Please train the model first.")
    except Exception as e:
        logging.error(f"Error loading the model: {e}")
        raise

# Function to predict review
def predict_review(review):
    logging.debug(f"Received review for prediction: {review}")
    try:
        cleaned_review = preprocess_text(review)
        logging.debug(f"Cleaned review: {cleaned_review}")

        tfidf_features = tfidf_vectorizer.transform([cleaned_review])
        tfidf_features_pca = pca.transform(tfidf_features.toarray())

        probabilities = stacking_model.predict_proba(tfidf_features_pca)[0]
        prob_genuine = probabilities[0]
        prob_fake = probabilities[1]

        prediction = "Fake" if prob_fake > prob_genuine else "Genuine"
        confidence = round(max(prob_genuine, prob_fake) * 100, 2)

        logging.debug(f"Prediction: {prediction}, Confidence: {confidence}")
        return prediction, confidence
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise

# API route for predicting reviews
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        review = data.get('review', '')
        if not review:
            logging.error("No review provided.")
            return jsonify({'error': 'No review provided'}), 400

        # Load model if not already loaded
        load_trained_model()

        prediction, confidence = predict_review(review)
        gc.collect()

        return jsonify({
            'prediction': prediction,
            'confidence': confidence
        })
    except Exception as e:
        logging.error(f"Error in /analyze route: {e}")
        return jsonify({'error': str(e)}), 500

# Function to evaluate the model
def evaluate_model(reviews, labels):
    try:
        logging.debug("Evaluating the model...")
        cleaned_reviews = reviews.apply(preprocess_text)
        tfidf_features = tfidf_vectorizer.transform(cleaned_reviews)
        tfidf_features_pca = pca.transform(tfidf_features.toarray())

        y_pred = stacking_model.predict(tfidf_features_pca)

        # Calculate evaluation metrics
        cm = confusion_matrix(labels, y_pred)
        logging.debug(f"Confusion Matrix:\n{cm}")

        report = classification_report(labels, y_pred, target_names=["Genuine", "Fake"])
        logging.debug(f"Classification Report:\n{report}")

        logging.debug("Evaluation complete.")
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        raise

# Main application start point
if __name__ == "__main__":
    # Load reviews and labels from CSV
    df = load_reviews('reviews.csv')
    
    # Split data for training and testing
    reviews = df['review']
    labels = df['label']

    test_size = min(200, len(reviews) // 5)  # Set the test size to 200 or fraction of data
    X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=test_size, random_state=42)

    # Train the model
    train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(X_test, y_test)

    # Run the Flask app, ensuring dynamic port allocation
    port = int(os.environ.get("PORT", 5000))  # Use the PORT environment variable or default to 5000
    app.run(host='0.0.0.0', port=port)
