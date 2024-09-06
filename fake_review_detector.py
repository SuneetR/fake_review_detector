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
from sklearn.decomposition import PCA
import string
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import joblib

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

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

# Load dataset
data = pd.read_csv('reviews.csv')  
data['label'] = data['label'].map({'fake': 0, 'genuine': 1})

# Apply preprocessing
data['cleaned_review'] = data['review'].apply(preprocess_text)

# Feature engineering
data['review_length'] = data['cleaned_review'].apply(lambda x: len(x.split()))
data['word_diversity'] = data['cleaned_review'].apply(lambda x: len(set(x.split())) / len(x.split()) if len(x.split()) > 0 else 0)

# Split data
X = data['cleaned_review']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sentence-BERT embeddings
model_name = "sentence-transformers/paraphrase-albert-small-v2"
sbert_model = SentenceTransformer(model_name)

def get_sbert_embeddings(text_data, batch_size=32):
    embeddings = []
    for i in range(0, len(text_data), batch_size):
        batch_embeddings = sbert_model.encode(text_data[i:i + batch_size].tolist(), convert_to_numpy=True)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# Generate embeddings
X_train_sbert = get_sbert_embeddings(X_train)
X_test_sbert = get_sbert_embeddings(X_test)

# Reduce SBERT dimensionality with PCA (Reduced to 32 dimensions)
pca = PCA(n_components=32)  
X_train_sbert = pca.fit_transform(X_train_sbert)
X_test_sbert = pca.transform(X_test_sbert)

# TF-IDF with reduced max_features to 200 and dtype=float16 for memory efficiency
tfidf_vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1, 2), dtype=np.float16)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Review length and word diversity to sparse matrices
X_train_meta = csr_matrix(X_train[['review_length', 'word_diversity']].values, dtype=np.float16)
X_test_meta = csr_matrix(X_test[['review_length', 'word_diversity']].values, dtype=np.float16)

# Combine features into sparse matrices for memory efficiency
X_train_combined = hstack([csr_matrix(X_train_tfidf), csr_matrix(X_train_sbert, dtype=np.float16), X_train_meta], format='csr')
X_test_combined = hstack([csr_matrix(X_test_tfidf), csr_matrix(X_test_sbert, dtype=np.float16), X_test_meta], format='csr')

# Base models with smaller XGB and Logistic Regression
lr = LogisticRegression(max_iter=2000, class_weight='balanced', C=0.1)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=20, max_depth=1, learning_rate=0.05)

# Stacking model
stacking_model = StackingClassifier(estimators=[
    ('lr', lr),
    ('xgb', xgb)
], final_estimator=LogisticRegression(), cv=StratifiedKFold(n_splits=5))

# Train the model
stacking_model.fit(X_train_combined, y_train)

# Predict and evaluate
y_pred = stacking_model.predict(X_test_combined)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Optimized F1 score: {f1:.4f}")

# Save the trained model, vectorizer, and PCA
joblib.dump(stacking_model, 'optimized_fake_review_detector_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(pca, 'pca.pkl')
