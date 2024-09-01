import string
import nltk
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
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

# Load data from CSV file (Ensure the file exists and is correctly placed)
try:
    df = pd.read_csv('reviews.csv')
except FileNotFoundError:
    raise Exception("The file 'reviews.csv' was not found. Please make sure it's in the correct directory.")

# Data preprocessing function
def preprocess_text(text):
    text = text.lower()  # Lowercase text
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])  # Remove stopwords and lemmatize
    return text

# Apply preprocessing to the dataset
df['review'] = df['review'].apply(preprocess_text)

# Ensure 'label' column is numeric (1 for fake, 0 for genuine)
if not pd.api.types.is_numeric_dtype(df['label']):
    df['label'] = df['label'].astype(int)

# Create a custom transformer to extract the product type
def product_type_extractor(X):
    return X[['product_type']]

# Create a pipeline with separate paths for the product type and review text
pipeline = make_pipeline(
    ColumnTransformer([
        ('text', TfidfVectorizer(), 'review'),
        ('product', FunctionTransformer(product_type_extractor), 'product_type')
    ]),
    LogisticRegression(solver='liblinear', random_state=42)
)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'columntransformer__text__ngram_range': [(1, 1), (1, 2)],  # Unigrams and bigrams
    'columntransformer__text__max_df': [0.75, 1.0],
    'columntransformer__text__min_df': [1, 5],
    'logisticregression__C': [0.1, 1, 10]  # Regularization strength
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(df[['review', 'product_type']], df['label'])

# Best model from GridSearch
best_model = grid_search.best_estimator_

# Calibrate the model to improve probability estimates
calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid', cv='prefit')
calibrated_model.fit(df[['review', 'product_type']], df['label'])

# Evaluate the model on the test set
X_train, X_test, y_train, y_test = train_test_split(df[['review', 'product_type']], df['label'], test_size=0.2, random_state=42)
y_pred = calibrated_model.predict(X_test)
y_proba = calibrated_model.predict_proba(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Function to predict if a review is fake or genuine
def predict_review(product_type, review):
    # Preprocess the review
    preprocessed_review = preprocess_text(review)
    # Predict
    prediction = calibrated_model.predict(pd.DataFrame({'product_type': [product_type], 'review': [preprocessed_review]}))[0]
    confidence = calibrated_model.predict_proba(pd.DataFrame({'product_type': [product_type], 'review': [preprocessed_review]}))[0].max() * 100
    # Interpret the result
    result = 'Fake' if prediction == 1 else 'Genuine'
    return result, f"{confidence:.2f}%"

# Self-learning function to update the model with new data
def update_model(product_type, new_review, new_label):
    global X_train, y_train
    # Preprocess the new review
    preprocessed_review = preprocess_text(new_review)
    # Append new data to the training set
    X_train = pd.concat([X_train, pd.DataFrame({'product_type': [product_type], 'review': [preprocessed_review]})], ignore_index=True)
    y_train = pd.concat([y_train, pd.Series(new_label)], ignore_index=True)
    # Retrain the model with the new data
    calibrated_model.fit(X_train, y_train)
