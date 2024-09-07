from flask import Flask, request, jsonify
import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
import string
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import gc
import logging

app = Flask(__name__)

# Download necessary NLTK data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the CSV file containing reviews
def load_reviews(file_path):
    return pd.read_csv(file_path)

# Tokenizer
def basic_tokenize(text):
    tokens = text.lower().split()
    tokens = [word.strip(string.punctuation) for word in tokens]
    return tokens

# Preprocessing
def preprocess_text(text):
    tokens = basic_tokenize(text)
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(cleaned_tokens)

# Sentence-BERT embeddings
model_name = "sentence-transformers/paraphrase-albert-small-v2"
sbert_model = SentenceTransformer(model_name)

def get_sbert_embeddings(text_data, batch_size=32):
    embeddings = []
    for i in range(0, len(text_data), batch_size):
        batch_embeddings = sbert_model.encode(text_data[i:i + batch_size].tolist(), convert_to_numpy=True)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# Model setup (No pre-trained models needed)
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
    # Preprocess reviews
    cleaned_reviews = reviews.apply(preprocess_text)
    
    # Generate Sentence-BERT embeddings
    sbert_embeddings = get_sbert_embeddings(cleaned_reviews)
    
    # Generate TF-IDF features
    tfidf_features = tfidf_vectorizer.fit_transform(cleaned_reviews)
    
    # Reduce dimensionality of embeddings using PCA
    sbert_embeddings_pca = pca.fit_transform(sbert_embeddings)
    
    # Combine features
    combined_features = np.hstack((tfidf_features.toarray(), sbert_embeddings_pca))
    
    # Train stacking classifier
    stacking_model.fit(combined_features, labels)

    # Free memory
    gc.collect()

# Prediction function
def predict_review(review):
    # Preprocess the review
    cleaned_review = preprocess_text(review)
    
    # Generate embeddings and transform features
    sbert_embedding = pca.transform(get_sbert_embeddings([cleaned_review]))
    tfidf_features = tfidf_vectorizer.transform([cleaned_review])
    
    # Combine features
    combined_features = np.hstack((tfidf_features.toarray(), sbert_embedding))
    
    # Get prediction probabilities
    probabilities = stacking_model.predict_proba(combined_features)[0]  # [prob_genuine, prob_fake]
    
    # Extract probabilities for each class
    prob_genuine = probabilities[0]  # Probability of genuine (class 0)
    prob_fake = probabilities[1]     # Probability of fake (class 1)
    
    # Assign labels based on higher probability
    if prob_fake > prob_genuine:
        prediction = "Fake"
        confidence = round(prob_fake * 100, 2)  # Convert to percentage with 2 decimal places
    else:
        prediction = "Genuine"
        confidence = round(prob_genuine * 100, 2)  # Convert to percentage with 2 decimal places
    
    # Return the prediction and confidence
    return prediction, confidence

# Route for predicting a single review
@app.route('/analyze', methods=['POST'])
def analyze():
    review = request.json.get('review')  # Get the review from the request
    prediction, confidence = predict_review(review)  # Use the model's predict_review function
    return jsonify({
        'prediction': prediction,
        'confidence': confidence
    })

# Evaluation function
def evaluate_model(reviews, labels):
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting evaluation...")
    
    # Preprocess reviews
    cleaned_reviews = reviews.apply(preprocess_text)
    
    # Generate Sentence-BERT embeddings
    sbert_embeddings = get_sbert_embeddings(cleaned_reviews)
    
    # Generate TF-IDF features
    tfidf_features = tfidf_vectorizer.transform(cleaned_reviews)
    
    # Reduce dimensionality of embeddings using PCA
    sbert_embeddings_pca = pca.transform(sbert_embeddings)
    
    # Combine features
    combined_features = np.hstack((tfidf_features.toarray(), sbert_embeddings_pca))
    
    # Predict on test set
    y_pred = stacking_model.predict(combined_features)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, y_pred)
    precision = precision_score(labels, y_pred)
    recall = recall_score(labels, y_pred)
    f1 = f1_score(labels, y_pred)
    conf_matrix = confusion_matrix(labels, y_pred)
    
    # Log metrics
    logging.info(f'Accuracy: {accuracy:.4f}')
    logging.info(f'Precision: {precision:.4f}')
    logging.info(f'Recall: {recall:.4f}')
    logging.info(f'F1 Score: {f1:.4f}')
    
    # Print confusion matrix in the logs
    logging.info(f'Confusion Matrix:\n{conf_matrix}')
    
    return accuracy, precision, recall, f1, conf_matrix

# Example usage (integrated pipeline)
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
    
    # Run Flask app
    app.run(debug=True)
