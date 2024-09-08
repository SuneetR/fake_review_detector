import logging
import gc
import string
import nltk
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Ensure required NLTK data is downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = Flask(__name__)

# Function to load CSV reviews
def load_reviews(file_path):
    logging.debug(f"Loading reviews from {file_path}")
    return pd.read_csv(file_path)

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

# Load the sentence-transformer model
model_name = "sentence-transformers/paraphrase-albert-small-v2"
sbert_model = SentenceTransformer(model_name)

# Get embeddings
def get_sbert_embeddings(text_data, batch_size=32):
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

# Model setup
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
pca = PCA(n_components=100)
stacking_model = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ],
    final_estimator=LogisticRegression()
)

# Model training function
def train_model(reviews, labels):
    try:
        logging.debug("Starting model training...")
        cleaned_reviews = reviews.apply(preprocess_text)
        sbert_embeddings = get_sbert_embeddings(cleaned_reviews)
        tfidf_features = tfidf_vectorizer.fit_transform(cleaned_reviews)
        sbert_embeddings_pca = pca.fit_transform(sbert_embeddings)
        combined_features = np.hstack((tfidf_features.toarray(), sbert_embeddings_pca))

        stacking_model.fit(combined_features, labels)
        logging.debug("Model training complete.")
        gc.collect()
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

# Prediction function
def predict_review(review):
    logging.debug(f"Received review for prediction: {review}")
    
    try:
        cleaned_review = preprocess_text(review)
        logging.debug(f"Cleaned review: {cleaned_review}")

        sbert_embedding = pca.transform(get_sbert_embeddings([cleaned_review]))
        tfidf_features = tfidf_vectorizer.transform([cleaned_review])
        combined_features = np.hstack((tfidf_features.toarray(), sbert_embedding))

        probabilities = stacking_model.predict_proba(combined_features)[0]
        prob_genuine = probabilities[0]
        prob_fake = probabilities[1]

        prediction = "Fake" if prob_fake > prob_genuine else "Genuine"
        confidence = round(max(prob_genuine, prob_fake) * 100, 2)

        logging.debug(f"Prediction: {prediction}, Confidence: {confidence}")
        return prediction, confidence
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise

# Flask route to handle review analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        review = data.get('review', '')
        if not review:
            return jsonify({'error': 'No review provided'}), 400

        prediction, confidence = predict_review(review)
        return jsonify({
            'prediction': prediction,
            'confidence': confidence
        })
    except Exception as e:
        logging.error(f"Error in /analyze: {e}")
        return jsonify({'error': str(e)}), 500

# Model evaluation function
def evaluate_model(reviews, labels):
    try:
        logging.debug("Evaluating the model...")
        cleaned_reviews = reviews.apply(preprocess_text)
        sbert_embeddings = get_sbert_embeddings(cleaned_reviews)
        tfidf_features = tfidf_vectorizer.transform(cleaned_reviews)
        sbert_embeddings_pca = pca.transform(sbert_embeddings)
        combined_features = np.hstack((tfidf_features.toarray(), sbert_embeddings_pca))

        y_pred = stacking_model.predict(combined_features)
        
        # Calculate and log evaluation metrics here if necessary
        logging.debug("Evaluation complete.")
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    # Load reviews and labels from CSV
    df = load_reviews('reviews.csv')

    # Assuming the CSV has two columns: 'review' and 'label' (0 or 1 for fake/real)
    reviews = df['review']
    labels = df['label']

    # Split the data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)

    # Train the model
    train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(X_test, y_test)

    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)
