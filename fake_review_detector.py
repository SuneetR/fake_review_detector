# fake_review_detector.py

import string
import nltk
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
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

# Preprocess function
def preprocess_text(text):
    text = text.lower()  # Lowercase text
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])  # Remove stopwords and lemmatize
    return text

# Load and split data
df = load_and_preprocess_data()
X = df['review']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model pipeline
pipeline = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression(solver='liblinear', C=1.0, random_state=42)
)

# Train the initial model
pipeline.fit(X_train, y_train)

# Predict function for new reviews
def predict_review(review):
    preprocessed_review = preprocess_text(review)
    prediction = pipeline.predict([preprocessed_review])[0]
    confidence = pipeline.predict_proba([preprocessed_review])[0].max() * 100
    result = 'True' if prediction == 1 else 'Fake'
    return result, f"{confidence:.2f}%"

# Function to update the model with a new review and label
def update_model(new_review, new_label):
    global X_train, y_train
    preprocessed_review = preprocess_text(new_review)
    
    # Append new data to the training set
    X_train = pd.concat([X_train, pd.Series(preprocessed_review)], ignore_index=True)
    y_train = pd.concat([y_train, pd.Series(new_label)], ignore_index=True)
    
    # Retrain the model on the updated dataset
    pipeline.fit(X_train, y_train)
