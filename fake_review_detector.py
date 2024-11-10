import os
import pickle
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

# Train the model and save components
def train_and_save_model():
    # Load the data
    X, y = load_data()

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

    # Evaluate the model
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    stacking_model.fit(X_train, y_train)
    y_pred = stacking_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    # Save the vectorizer, PCA, and model to disk
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    with open('pca.pkl', 'wb') as f:
        pickle.dump(pca, f)
    with open('stacking_model.pkl', 'wb') as f:
        pickle.dump(stacking_model, f)

    print("Model components saved successfully.")

if __name__ == "__main__":
    train_and_save_model()
