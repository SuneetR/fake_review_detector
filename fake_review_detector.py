# fake_review_detector.py

import string
import nltk
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load and preprocess the data from CSV
def load_and_preprocess_data(file_path='reviews.csv'):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise Exception("The file 'reviews.csv' was not found. Please make sure it's in the correct directory.")
    
    # Ensure no missing values in 'label' column and preprocess text
    df.dropna(subset=['label'], inplace=True)
    df['review'] = df['review'].apply(preprocess_text)
    
    # Ensure 'label' is numeric (1 for true, 0 for fake)
    df['label'] = df['label'].astype(int)
    
    return df

# Custom preprocessing function for text data
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Custom transformer to calculate review length and adjective count
class LengthAndAdjectiveTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        features = pd.DataFrame()
        
        # Length of the review
        features['review_length'] = X['review'].apply(lambda x: len(x.split()))
        
        # Adjective count
        features['adjective_count'] = X['review'].apply(lambda x: sum(1 for word in x.split() if word.endswith('y')))
        
        return features

# Load data and split into train/test
df = load_and_preprocess_data()
X = df[['review']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a column transformer with both text vectorization and custom features
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=5), 'review'),
        ('custom_features', LengthAndAdjectiveTransformer(), ['review'])
    ]
)

# Pipeline with feature extraction and model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', C=1.0, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Prediction function
def predict_review(review):
    # Convert to DataFrame for compatibility with ColumnTransformer
    X_input = pd.DataFrame({'review': [preprocess_text(review)]})
    prediction = pipeline.predict(X_input)[0]
    confidence = pipeline.predict_proba(X_input)[0].max() * 100
    result = 'True' if prediction == 1 else 'Fake'
    return result, f"{confidence:.2f}%"

# Function to update the model with new data
def update_model(new_review, new_label):
    global X_train, y_train
    
    # Preprocess and append new data to training set
    new_review_processed = preprocess_text(new_review)
    X_train = pd.concat([X_train, pd.DataFrame({'review': [new_review_processed]})], ignore_index=True)
    y_train = pd.concat([y_train, pd.Series([new_label])], ignore_index=True)
    
    # Retrain the model with updated data
    pipeline.fit(X_train, y_train)
