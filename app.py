import os
import logging
import gc
import string
import nltk
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from flask import Flask, request, jsonify, render_template

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Ensure required NLTK data is downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Initialize Flask app
app = Flask(__name__)

# Tokenizer function
def basic_tokenize(text):
    tokens = text.lower().split()
    tokens = [word.strip(string.punctuation) for word in tokens]
    return tokens

# Preprocessing function
def preprocess_text(text):
    tokens = basic_tokenize(text)
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(cleaned_tokens)

# Function to predict a review
def predict_review(review):
    try:
        logging.debug(f"Received review for prediction: {review}")
        cleaned_review = preprocess_text(review)
        logging.debug(f"Cleaned review: {cleaned_review}")

        # Ensure all model components are loaded
        if not all([tfidf_vectorizer, pca, stacking_model]):
            raise RuntimeError("Model components (TF-IDF, PCA, or classifier) are not properly loaded.")

        # Generate TF-IDF features and PCA-reduced features
        tfidf_features = tfidf_vectorizer.transform([cleaned_review])
        tfidf_features_pca = pca.transform(tfidf_features.toarray())

        # Predict using the stacking model
        probabilities = stacking_model.predict_proba(tfidf_features_pca)[0]
        prob_genuine = probabilities[0]
        prob_fake = probabilities[1]

        # Determine prediction and confidence
        prediction = "Fake" if prob_fake > prob_genuine else "Genuine"
        confidence = round(max(prob_genuine, prob_fake) * 100, 2)

        logging.debug(f"Prediction: {prediction}, Confidence: {confidence}")
        return prediction, confidence
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise

# Route for Home Page (Serve index.html)
@app.route('/')
def home():
    return render_template('index.html')

# API route for analyzing review sentiment
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Parse the input data
        data = request.get_json()
        review = data.get('review', '')
        if not review:
            logging.error("No review provided.")
            return jsonify({'error': 'No review provided'}), 400

        # Make the prediction
        prediction, confidence = predict_review(review)
        gc.collect()

        # Return the result as JSON
        return jsonify({
            'prediction': prediction,
            'confidence': confidence
        })
    except Exception as e:
        logging.error(f"Error in /analyze route: {e}")
        return jsonify({'error': str(e)}), 500

# Start the Flask application
if __name__ == "__main__":
    # Start the Flask app on a dynamic port
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
