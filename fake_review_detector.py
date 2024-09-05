import string
import nltk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocess text function
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Load data from CSV file
try:
    df = pd.read_csv('reviews.csv')
except FileNotFoundError:
    raise Exception("The file 'reviews.csv' was not found. Please make sure it's in the correct directory.")

# Apply preprocessing to the dataset
df['review'] = df['review'].apply(preprocess_text)

# Ensure 'label' column is numeric (1 for genuine, 0 for fake)
if not pd.api.types.is_numeric_dtype(df['label']):
    df['label'] = df['label'].astype(int)

# Split the dataset into features and target
X = df['review']
y = df['label']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature engineering pipeline using TF-IDF and optional dimensionality reduction
feature_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 1))),  # Reduce features
    ('svd', TruncatedSVD(n_components=50))  # Reduce SVD components
])

# Transform the features
X_train_transformed = feature_pipeline.fit_transform(X_train)
X_test_transformed = feature_pipeline.transform(X_test)

# Handle class imbalance using SMOTE on numerical features
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)

# XGBoost Classifier (further reduced parameters)
xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42, n_estimators=30, max_depth=3)  # Reduce complexity

# Hyperparameter tuning using GridSearchCV (reduced grid size)
param_grid = {
    'learning_rate': [0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

grid_search = GridSearchCV(xgb_model, param_grid, cv=2, n_jobs=-1, scoring='roc_auc')  # Reduce cross-validation folds
grid_search.fit(X_train_resampled, y_train_resampled)

# Best model from GridSearch
best_model = grid_search.best_estimator_

# Calibrate the model to improve probability estimates
calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid', cv=2)  # Reduce cross-validation folds
calibrated_model.fit(X_train_resampled, y_train_resampled)

# Evaluate the model on the test set
y_pred = calibrated_model.predict(X_test_transformed)
y_proba = calibrated_model.predict_proba(X_test_transformed)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba[:, 1]))

# Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba[:, 1])
print(f'Precision: {precision}\nRecall: {recall}')

# Function to predict if a review is fake or genuine
def predict_review(review):
    # Preprocess the review
    preprocessed_review = preprocess_text(review)
    # Transform the review to features
    transformed_review = feature_pipeline.transform([preprocessed_review])
    # Predict
    prediction = calibrated_model.predict(transformed_review)[0]
    confidence = calibrated_model.predict_proba(transformed_review)[0].max() * 100
    # Interpret the result
    result = 'Genuine' if prediction == 1 else 'Fake'
    return result, f"{confidence:.2f}%"

# Self-learning function to update the model with new data
def update_model(new_review, new_label):
    global X_train_resampled, y_train_resampled
    # Preprocess the new review
    preprocessed_review = preprocess_text(new_review)
    # Transform the new review to features
    transformed_review = feature_pipeline.transform([preprocessed_review])
    # Append new data to the training set
    X_train_resampled = np.vstack([X_train_resampled, transformed_review])
    y_train_resampled = np.append(y_train_resampled, new_label)
    # Retrain the model with the new data
    best_model.fit(X_train_resampled, y_train_resampled)
    # Recalibrate the model
    global calibrated_model
    calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid', cv=2)
    calibrated_model.fit(X_train_resampled, y_train_resampled)
