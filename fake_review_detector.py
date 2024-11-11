# Enhanced fake_review_detector.py

import string
import nltk
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# Initialize NLTK components
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Enhanced Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Custom Transformer
class AdvancedFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        features = pd.DataFrame()
        
        # Review Length
        features['review_length'] = X['review'].apply(lambda x: len(x.split()))
        
        # Lexical Diversity
        features['lexical_diversity'] = X['review'].apply(lambda x: len(set(x.split())) / len(x.split()) if len(x.split()) > 0 else 0)
        
        # Sentiment Polarity and Subjectivity
        features['sentiment_polarity'] = X['review'].apply(lambda x: TextBlob(x).sentiment.polarity)
        features['sentiment_subjectivity'] = X['review'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
        
        # Specificity (based on presence of named entities, here estimated by nouns as a simple proxy)
        features['noun_count'] = X['review'].apply(lambda x: sum(1 for word in x.split() if word.endswith('y')))
        
        return features

# Load data and preprocess
def load_and_preprocess_data(file_path='reviews.csv'):
    df = pd.read_csv(file_path)
    df.dropna(subset=['label'], inplace=True)
    df['review'] = df['review'].apply(preprocess_text)
    df['label'] = df['label'].astype(int)
    return df

# Load and split data
df = load_and_preprocess_data()
X = df[['review']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Column Transformer and Model Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=5), 'review'),
        ('custom_features', AdvancedFeatureExtractor(), ['review'])
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42, n_estimators=100))
])

# Train the model
pipeline.fit(X_train, y_train)

# Prediction function
def predict_review(review):
    X_input = pd.DataFrame({'review': [preprocess_text(review)]})
    prediction = pipeline.predict(X_input)[0]
    confidence = pipeline.predict_proba(X_input)[0].max() * 100
    result = 'True' if prediction == 1 else 'Fake'
    return result, f"{confidence:.2f}%"
