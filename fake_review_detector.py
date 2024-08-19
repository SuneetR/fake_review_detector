import string
import nltk
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.utils import shuffle

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load data from CSV file
df = pd.read_csv('reviews.csv')

# Data preprocessing function
def preprocess_text(text):
    text = text.lower()  # Lowercase text
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])  # Remove stopwords and lemmatize
    return text

# Apply preprocessing to the dataset
df['review'] = df['review'].apply(preprocess_text)

# Shuffle the dataset to ensure randomness
df = shuffle(df, random_state=42)

# Split the dataset into training and test sets
X = df['review']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with TfidfVectorizer and LogisticRegression
pipeline = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression(solver='liblinear', C=1.0, random_state=42)  # Using liblinear solver for small datasets
)

# Train the model with cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print(f'Cross-validation accuracy: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}')

pipeline.fit(X_train, y_train)

# Save the model to disk
joblib.dump(pipeline, 'review_model.pkl')

# Load the model (this would be used when deploying the model)
pipeline = joblib.load('review_model.pkl')

# Function to predict if a review is fake or genuine
def predict_review(review):
    # Preprocess the review
    preprocessed_review = preprocess_text(review)
    # Predict
    prediction = pipeline.predict([preprocessed_review])[0]
    confidence = pipeline.predict_proba([preprocessed_review])[0].max() * 100
    # Interpret the result
    result = 'Fake' if prediction == 1 else 'Genuine'
    return result, f"{confidence:.2f}%"

# Self-learning function to update the model with new data
def update_model(new_review, new_label):
    global X_train, y_train
    # Preprocess the new review
    preprocessed_review = preprocess_text(new_review)
    # Append new data to the training set
    X_train = X_train.append(pd.Series(preprocessed_review), ignore_index=True)
    y_train = y_train.append(pd.Series(new_label), ignore_index=True)
    # Retrain the model
    pipeline.fit(X_train, y_train)
    # Save the updated model
    joblib.dump(pipeline, 'review_model.pkl')

# Example usage
print(predict_review("This product is amazing!"))
update_model("Worst experience ever!", 1)
