import string
import nltk
import spacy
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load Spacy model for more advanced NLP processing
nlp = spacy.load('en_core_web_sm')

# Load data from CSV file
try:
    df = pd.read_csv('reviews.csv')
except FileNotFoundError:
    raise Exception("The file 'reviews.csv' was not found. Please make sure it's in the correct directory.")

# Data preprocessing function with Spacy for better tokenization and lemmatization
def preprocess_text(text):
    doc = nlp(text.lower())  # Lowercase text and tokenize with Spacy
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.text not in stop_words]  # Remove stopwords and lemmatize
    return ' '.join(tokens)

# Apply preprocessing to the dataset
df['review'] = df['review'].apply(preprocess_text)

# Ensure 'label' column is numeric (1 for genuine, 0 for fake)
if not pd.api.types.is_numeric_dtype(df['label']):
    df['label'] = df['label'].astype(int)

# Split the dataset into features and target
X = df['review']
y = df['label']

# Custom transformer to extract additional features
class CustomFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, product_keywords=None):
        self.product_keywords = product_keywords if product_keywords else {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for review in X:
            doc = nlp(review)
            keyword_count = sum([1 for token in doc if token.text in self.product_keywords])
            review_length = len(doc)
            features.append([keyword_count, review_length])
        return np.array(features)

# Define product-specific keywords (you can update these)
product_keywords = {
    'battery': ['battery', 'charge', 'power', 'capacity'],
    'mobile': ['phone', 'mobile', 'smartphone', 'android', 'iphone'],
    'book': ['book', 'author', 'story', 'plot', 'character'],
    # Add more product categories and keywords as needed
}

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with TfidfVectorizer and RandomForestClassifier (better for handling complex patterns)
pipeline = make_pipeline(
    FeatureUnion([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_df=0.75, min_df=5)),  # You can experiment with these parameters
        ('custom_features', CustomFeatures(product_keywords=product_keywords)),
    ]),
    RandomForestClassifier(n_estimators=100, random_state=42)  # Experiment with different models/hyperparameters
)

# Hyperparameter tuning using GridSearchCV (optional, to further fine-tune)
param_grid = {
    'randomforestclassifier__n_estimators': [100, 200],
    'randomforestclassifier__max_depth': [None, 10, 20],
    'randomforestclassifier__min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model from GridSearch
best_model = grid_search.best_estimator_

# Calibrate the model to improve probability estimates
calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid', cv=5)
calibrated_model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = calibrated_model.predict(X_test)
y_proba = calibrated_model.predict_proba(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Function to predict if a review is fake or genuine
def predict_review(review):
    # Preprocess the review
    preprocessed_review = preprocess_text(review)
    # Predict
    prediction = calibrated_model.predict([preprocessed_review])[0]
    confidence = calibrated_model.predict_proba([preprocessed_review])[0].max() * 100
    # Interpret the result
    result = 'Genuine' if prediction == 1 else 'Fake'
    return result, f"{confidence:.2f}%"

# Self-learning function to update the model with new data
def update_model(new_review, new_label):
    global X_train, y_train
    # Preprocess the new review
    preprocessed_review = preprocess_text(new_review)
    # Append new data to the training set
    X_train = pd.concat([X_train, pd.Series(preprocessed_review)], ignore_index=True)
    y_train = pd.concat([y_train, pd.Series(new_label)], ignore_index=True)
    # Retrain the model with the new data
    pipeline.fit(X_train, y_train)
    # Recalibrate the model
    global calibrated_model
    calibrated_model = CalibratedClassifierCV(pipeline, method='sigmoid', cv=5)
    calibrated_model.fit(X_train, y_train)
   
