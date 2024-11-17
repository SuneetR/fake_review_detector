import string
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import gc

# Load pretrained lightweight ALBERT v2 model
tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
albert_model = AutoModel.from_pretrained("albert-base-v2")

# Custom Transformer using ALBERT
class ALBERTFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, model, tokenizer, max_length=128):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        features = []
        for text in X['review']:
            # Tokenize and truncate input text
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
            
            # Disable gradient calculations for inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use pooled output ([CLS] token representation)
            pooled_output = outputs.last_hidden_state[:, 0, :].numpy()
            features.append(pooled_output.flatten())
        
        # Convert features to DataFrame
        return pd.DataFrame(features)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

# Load and preprocess data
def load_data(file_path='reviews.csv'):
    df = pd.read_csv(file_path)
    df.dropna(subset=['label'], inplace=True)
    df['review'] = df['review'].apply(preprocess_text)
    df['label'] = df['label'].astype(int)
    return df

# Load dataset
df = load_data()
X = df[['review']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Column Transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('albert_features', ALBERTFeatureExtractor(albert_model, tokenizer), ['review']),
        ('tfidf', TfidfVectorizer(max_features=500), 'review')  # Backup features
    ]
)

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42))  # Reduced complexity
])

# Fit Model
pipeline.fit(X_train, y_train)

# Prediction
def predict_review(review):
    review_processed = pd.DataFrame({'review': [preprocess_text(review)]})
    result = pipeline.predict(review_processed)
    return "True" if result[0] == 1 else "Fake"

# Memory Cleanup
gc.collect()
