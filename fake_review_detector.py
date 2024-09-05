import string
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
from transformers import BertTokenizer, BertModel
from sklearn.base import BaseEstimator, TransformerMixin
import gc

# Download NLTK resources if not already available
def download_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords.zip')
        nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('wordnet')

download_nltk_resources()

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = BertModel.from_pretrained('distilbert-base-uncased')

# Preprocess text function
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Convert text to BERT embeddings in batches
def text_to_bert_batch(texts):
    tokens = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128, is_split_into_words=False)
    with torch.no_grad():
        outputs = bert_model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Custom transformer for BERT embeddings
class BERTVectorizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        batch_size = 32  # Adjust batch size based on memory constraints
        embeddings = []
        for i in range(0, len(X), batch_size):
            batch_texts = X[i:i+batch_size]
            batch_embeddings = text_to_bert_batch(batch_texts)
            embeddings.append(batch_embeddings)
        return np.concatenate(embeddings)

# Load data from CSV file in chunks
def process_chunk(chunk):
    chunk['review'] = chunk['review'].apply(preprocess_text)
    return chunk

chunk_size = 10000  # Adjust chunk size based on memory constraints
chunks = pd.read_csv('reviews.csv', chunksize=chunk_size)
df = pd.concat(process_chunk(chunk) for chunk in chunks)

# Ensure 'label' column is numeric (1 for genuine, 0 for fake)
if not pd.api.types.is_numeric_dtype(df['label']):
    df['label'] = df['label'].astype(int)

# Split the dataset into features and target
X = df['review']
y = df['label']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Advanced feature engineering pipeline
feature_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),  # Reduced features
    ('pca', PCA(n_components=50))  # Fewer components
])

# Combine feature extraction with SMOTE and XGBoost in an imbalanced pipeline
pipeline = ImbPipeline([
    ('features', feature_pipeline),
    ('smote', SMOTE(random_state=42)),
    ('xgb', xgb.XGBClassifier(objective='binary:logistic', random_state=42))
])

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'xgb__learning_rate': [0.01, 0.05],
    'xgb__max_depth': [6, 10],
    'xgb__subsample': [0.7, 0.8],
    'xgb__colsample_bytree': [0.7, 0.8],
    'xgb__n_estimators': [100, 200]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring='f1')
grid_search.fit(X_train, y_train)

# Best model from GridSearch
best_model = grid_search.best_estimator_

# Evaluate the model on the test set
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba[:, 1]))

# Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba[:, 1])
print(f'Precision: {precision}\nRecall: {recall}')

# Save the best model and vectorizer if needed
import joblib
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(feature_pipeline, 'feature_pipeline.pkl')

# Release memory
del df, X, y, X_train, X_test, y_train, y_test
gc.collect()
