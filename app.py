import os
import logging
import gc
import string
import nltk
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
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

# Load sentence-transformer model
model_name = "sentence-transformers/paraphrase-albert-small-v2"
sbert_model = SentenceTransformer(model_name)

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

# Function to get embeddings using SBERT
def get_sbert_embeddings(text_data, batch_size=4):
    try:
        logging.debug(f"Generating SBERT embeddings for {len(text_data)} reviews.")
        embeddings = []
        for i in range(0, len(text_data), batch_size):
            batch_embeddings = sbert_model.encode(text_data[i:i + batch_size].tolist(), convert_to_numpy=True)
            embeddings.append(batch_embeddings)
        return np.vstack(embeddings)
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        raise

# Load trained model and vectorizer
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
            raise FileNotFoundError("Trained model file not found.")
    except Exception as e:
        logging.error(f"Error loading the model: {e}")
        raise

# Function to predict a review
def predict_review(review):
    try:
        logging.debug(f"Received review for prediction: {review}")
        cleaned_review = preprocess_text(review)
        logging.debug(f"Cleaned review: {cleaned_review}")

        # Generate SBERT embedding and TF-IDF features
        sbert_embedding = pca.transform(get_sbert_embeddings([cleaned_review]))
        tfidf_features = tfidf_vectorizer.transform([cleaned_review])
        combined_features = np.hstack((tfidf_features.toarray(), sbert_embedding))

        # Predict using the stacking model
        probabilities = stacking_model.predict_proba(combined_features)[0]
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

        # Load model if not already loaded
        load_trained_model()

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
    # Load model when the server starts
    load_trained_model()

    # Start the Flask app on a dynamic port
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
