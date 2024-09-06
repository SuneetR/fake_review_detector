import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from scipy.sparse import hstack, csr_matrix
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import joblib

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

# Load the dataset
data = pd.read_csv('reviews.csv')  # Replace with your dataset path
data['label'] = data['label'].map({'fake': 0, 'genuine': 1})  # Adjust according to your dataset

# Preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    cleaned_tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return ' '.join(cleaned_tokens)

# Apply preprocessing
data['cleaned_review'] = data['review'].apply(preprocess_text)

# Feature engineering: review length and word diversity
data['review_length'] = data['cleaned_review'].apply(lambda x: len(x.split()))
data['word_diversity'] = data['cleaned_review'].apply(lambda x: len(set(x.split())) / len(x.split()) if len(x.split()) > 0 else 0)

# Split data
X = data['cleaned_review']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sentence-BERT embeddings with a smaller model
model_name = "sentence-transformers/paraphrase-albert-small-v2"  # Smaller model with 128-dimensional vectors
sbert_model = SentenceTransformer(model_name)

# Batch processing for SBERT embeddings
def get_sbert_embeddings(text_data, batch_size=32):
    embeddings = []
    for i in range(0, len(text_data), batch_size):
        batch_embeddings = sbert_model.encode(text_data[i:i + batch_size].tolist(), convert_to_numpy=True)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# Generate embeddings in batches to reduce memory consumption
X_train_sbert = get_sbert_embeddings(X_train)
X_test_sbert = get_sbert_embeddings(X_test)

# TF-IDF Vectorization with reduced features and dtype=float32 for efficiency
tfidf_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), dtype=np.float32)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Combine features into sparse matrices for memory efficiency
X_train_combined = hstack([csr_matrix(X_train_tfidf), csr_matrix(X_train_sbert), csr_matrix(X_train[['review_length', 'word_diversity']].values)], format='csr')
X_test_combined = hstack([csr_matrix(X_test_tfidf), csr_matrix(X_test_sbert), csr_matrix(X_test[['review_length', 'word_diversity']].values)], format='csr')

# Define base models with optimized parameters
lr = LogisticRegression(max_iter=2000, class_weight='balanced', C=0.1)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=50, max_depth=2, learning_rate=0.05)

# Stacking model with fewer estimators
stacking_model = StackingClassifier(estimators=[
    ('lr', lr),
    ('xgb', xgb)
], final_estimator=LogisticRegression(), cv=StratifiedKFold(n_splits=5))

# Train the stacking model
stacking_model.fit(X_train_combined, y_train)

# Predict and evaluate
y_pred = stacking_model.predict(X_test_combined)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Optimized F1 score: {f1:.4f}")

# Save the trained model
joblib.dump(stacking_model, 'optimized_fake_review_detector_model.pkl')
