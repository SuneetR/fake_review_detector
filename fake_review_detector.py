import string
import spacy
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from textblob import TextBlob
import pandas as pd
import numpy as np

# Load SpaCy model
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Stopwords list
STOPWORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through"
])

# Load and preprocess the data from CSV
def load_and_preprocess_data(file_path='reviews.csv'):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise Exception("The file 'reviews.csv' was not found. Please make sure it's in the correct directory.")
    
    df.dropna(subset=['label'], inplace=True)
    df['review'] = df['review'].apply(preprocess_text)
    df['label'] = df['label'].astype(int)
    return df

# Custom preprocessing function for text data
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])
    return text

# Advanced Feature Extraction class
class AdvancedFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        features = pd.DataFrame()
        
        # Length of the review
        features['review_length'] = X['review'].apply(lambda x: len(x.split()))
        
        # Average word length
        features['avg_word_length'] = X['review'].apply(lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0)
        
        # Stopword ratio
        features['stopword_ratio'] = X['review'].apply(lambda x: sum(1 for word in x.split() if word in STOPWORDS) / len(x.split()) if x.split() else 0)
        
        # Noun to Verb ratio
        features['noun_verb_ratio'] = X['review'].apply(self.calculate_noun_verb_ratio)
        
        # Sentiment polarity and subjectivity
        features['sentiment_polarity'] = X['review'].apply(lambda x: TextBlob(x).sentiment.polarity)
        features['sentiment_subjectivity'] = X['review'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
        
        return features

    @staticmethod
    def calculate_noun_verb_ratio(text):
        doc = nlp(text)
        nouns = sum(1 for token in doc if token.pos_ == 'NOUN')
        verbs = sum(1 for token in doc if token.pos_ == 'VERB')
        return nouns / verbs if verbs > 0 else 0

# Load data and split into train/test
df = load_and_preprocess_data()
X = df[['review']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Column transformer with TF-IDF and custom features
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(ngram_range=(1, 2), max_df=0.8, min_df=3, max_features=10000), 'review'),
        ('custom_features', AdvancedFeatureTransformer(), ['review'])
    ]
)

# Pipeline with Logistic Regression
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'))
])

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'preprocessor__text__max_df': [0.8, 0.9],
    'preprocessor__text__min_df': [3, 5],
    'preprocessor__text__ngram_range': [(1, 1), (1, 2)]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Evaluate the best model
print(f"Best Parameters: {grid_search.best_params_}")
best_pipeline = grid_search.best_estimator_

# Cross-validation score
scores = cross_val_score(best_pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"Improved CV Accuracy: {np.mean(scores):.2f}")

# Prediction function
def predict_review(review):
    X_input = pd.DataFrame({'review': [preprocess_text(review)]})
    prediction = best_pipeline.predict(X_input)[0]
    confidence = best_pipeline.predict_proba(X_input)[0].max() * 100
    result = 'True' if prediction == 1 else 'Fake'
    return result, f"{confidence:.2f}%"

print("Model training and tuning complete.")
