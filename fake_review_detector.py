import os
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set up stop words
stop_words = {'a', 'the', 'is', 'in', 'it', 'to', 'and', 'of', 'on', 'for', 'with', 'as', 'by', 'an', 'at', 'or', 'that', 'this', 'which', 'be'}

# Preprocessing function
def preprocess_text(text):
    tokens = text.lower().split()
    tokens = [word.strip(string.punctuation) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load and preprocess dataset
def load_data(file_path='reviews.csv'):
    df = pd.read_csv(file_path)
    df['review'] = df['review'].apply(preprocess_text)  # Preprocess the review text
    return df['review'], df['label']

# Train the model
def train_model(X, y):
    # Vectorize text using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    X_tfidf = tfidf_vectorizer.fit_transform(X)

    # Reduce dimensionality with PCA
    pca = PCA(n_components=100)
    X_pca = pca.fit_transform(X_tfidf.toarray())

    # Define and train the model
    base_estimators = [('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))]
    stacking_model = StackingClassifier(estimators=base_estimators, final_estimator=LogisticRegression())
    stacking_model.fit(X_pca, y)

    return tfidf_vectorizer, pca, stacking_model

# Predict function
def predict(review, tfidf_vectorizer, pca, model):
    # Preprocess and vectorize input review
    review_processed = preprocess_text(review)
    review_vectorized = tfidf_vectorizer.transform([review_processed])
    review_pca = pca.transform(review_vectorized.toarray())

    # Get prediction and confidence
    prediction = model.predict(review_pca)[0]
    confidence = max(model.predict_proba(review_pca)[0]) * 100
    return prediction, confidence

if __name__ == "__main__":
    # Load and train the model
    X, y = load_data()
    tfidf_vectorizer, pca, model = train_model(X, y)
    
    # Evaluate the model
    X_train, X_test, y_train, y_test = train_test_split(tfidf_vectorizer.transform(X).toarray(), y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
