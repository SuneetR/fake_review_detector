import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import string

# Define stop words and preprocessing function
stop_words = {'a', 'the', 'is', 'in', 'it', 'to', 'and', 'of', 'on', 'for', 'with', 'as', 'by', 'an', 'at', 'or', 'that', 'this', 'which', 'be'}

def preprocess_text(text):
    tokens = text.lower().split()
    tokens = [word.strip(string.punctuation) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def load_data():
    # Load and preprocess data
    data = pd.read_csv('reviews.csv')
    data['review'] = data['review'].apply(preprocess_text)
    X = data['review']
    y = data['label']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    # Initialize components
    tfidf_vectorizer = TfidfVectorizer()
    pca = PCA(n_components=50)  # Adjust as needed
    base_models = [
        ('nb', MultinomialNB()),
        ('svc', SVC(probability=True))
    ]
    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression()
    )

    # Create a pipeline
    pipeline = Pipeline([
        ('tfidf', tfidf_vectorizer),
        ('pca', pca),
        ('stacking', stacking_model)
    ])

    # Fit model
    pipeline.fit(X_train, y_train)
    return pipeline

# Prepare data and train model on script execution
X_train, X_test, y_train, y_test = load_data()
model_pipeline = train_model(X_train, y_train)
