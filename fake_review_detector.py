import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from scipy.sparse import hstack, csr_matrix
from sklearn.decomposition import PCA
import string
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import joblib
import gc

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Lazy loaded models
stacking_model = None
tfidf_vectorizer = None
pca = None

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

# Load models
def load_models():
    global stacking_model, tfidf_vectorizer, pca
    if stacking_model is None:
        stacking_model = joblib.load('optimized_fake_review_detector_model.pkl')
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        pca = joblib.load('pca.pkl')

# Prediction function
def predict_review(review, product_type=None):
    load_models()
    
    # Preprocess review
    cleaned_review = preprocess_text(review)
    
    # Generate embeddings and transform features
    sbert_embedding = pca.transform(get_sbert_embeddings([cleaned_review]))
    tfidf_features = tfidf_vectorizer.transform([cleaned_review])
    
    # Combine features
    review_length = len(cleaned_review.split())
    word_diversity = len(set(cleaned_review.split())) / len(cleaned_review.split()) if len(cleaned_review.split()) > 0 else 0
    meta_features = csr_matrix([[review_length, word_diversity]], dtype=np.float16)
    
    combined_features = hstack([csr_matrix(tfidf_features), csr_matrix(sbert_embedding, dtype=np.float16), meta_features], format='csr')
    
    # Make predictions
    prediction = stacking_model.predict(combined_features)[0]
    confidence = max(stacking_model.predict_proba(combined_features)[0])
    
    # Free memory
    gc.collect()

    return prediction, confidence

# Model update function
def update_model(review, label):
    load_models()

    # Preprocess and append to the training dataset (or implement online learning)
    # This part requires a mechanism to retrain the model incrementally
    pass

# Evaluation function
def evaluate_model(X_test, y_test):
    load_models()
    
    # Predict on test set
    y_pred = stacking_model.predict(X_test)
    y_proba = stacking_model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Print metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Confusion Matrix:\n{conf_matrix}')

    return accuracy, precision, recall, f1, conf_matrix

# Example usage
if __name__ == "__main__":
    # Assuming you have your test data loaded as X_test and y_test
    # X_test = ... (Feature matrix for test data)
    # y_test = ... (True labels for test data)
    # evaluate_model(X_test, y_test)
    
    # Example for prediction
    review = "This product is amazing and I loved it!"
    prediction, confidence = predict_review(review)
    print(f'Prediction: {prediction}, Confidence: {confidence:.4f}')
