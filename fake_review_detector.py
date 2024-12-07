import string
import nltk
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLTK components
stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.WordNetLemmatizer()

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

# Feature transformer for advanced features
class EnhancedFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Load BERT model for sentiment analysis
        self.sentiment_model = hf_pipeline(
            "text-classification",
            model=AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment"),
            tokenizer=AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment"),
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        features = pd.DataFrame()
        
        # Length of the review
        features['review_length'] = X['review'].apply(lambda x: len(x.split()))
        
        # Lexical diversity
        features['lexical_diversity'] = X['review'].apply(lambda x: len(set(x.split())) / len(x.split()) if len(x.split()) > 0 else 0)
        
        # Average word length
        features['avg_word_length'] = X['review'].apply(lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0)
        
        # Sentiment analysis using BERT
        sentiment_scores = X['review'].apply(self.get_sentiment_score)
        features['bert_sentiment'] = sentiment_scores.apply(lambda x: x['score'])
        features['bert_sentiment_label'] = sentiment_scores.apply(lambda x: x['label'])
        
        return features

    def get_sentiment_score(self, text):
        # Use the BERT model to analyze sentiment
        result = self.sentiment_model(text[:512])[0]  # Truncate to 512 tokens for BERT
        return {"label": result['label'], "score": result['score']}

# Load data and split into train/test
df = load_and_preprocess_data()
X = df[['review']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a column transformer with both text vectorization and custom features
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=5), 'review'),
        ('custom_features', EnhancedFeatureTransformer(), ['review'])
    ]
)

# Pipeline with feature extraction and model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, max_depth=20))  # Increased depth
])

# Train the model
pipeline.fit(X_train, y_train)

# Prediction function
def predict_review(review, threshold=0.7):
    # Convert to DataFrame for compatibility with ColumnTransformer
    X_input = pd.DataFrame({'review': [preprocess_text(review)]})
    proba = pipeline.predict_proba(X_input)[0]
    prediction = 1 if proba[1] > threshold else 0
    confidence = proba.max() * 100
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
