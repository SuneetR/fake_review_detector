import string
import nltk
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import spacy
import joblib

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# Load SpaCy model
nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])

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

# Custom preprocessing function for text data
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Feature extraction transformer class
class AdvancedFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        features = pd.DataFrame()

        # Length of the review
        features['review_length'] = X['review'].apply(lambda x: len(x.split()))

        # Adjective count using POS tagging
        features['adjective_count'] = X['review'].apply(self.count_adjectives)

        # Lexical diversity
        features['lexical_diversity'] = X['review'].apply(lambda x: len(set(x.split())) / len(x.split()) if x.split() else 0)

        # Sentiment polarity and subjectivity
        features['sentiment_polarity'] = X['review'].apply(lambda x: TextBlob(x).sentiment.polarity)
        features['sentiment_subjectivity'] = X['review'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

        # Named entity count
        features['named_entity_count'] = X['review'].apply(self.count_named_entities)

        # First-person pronoun frequency
        features['first_person_pronouns'] = X['review'].apply(lambda x: sum(1 for word in x.split() if word in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']))

        # Emoji and exclamation marks
        features['emoji_count'] = X['review'].apply(lambda x: sum(1 for char in x if char in ['�', '�', '❤', '☺']))
        features['exclamation_count'] = X['review'].apply(lambda x: x.count('!'))

        return features

    @staticmethod
    def count_adjectives(text):
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        return sum(1 for word, tag in pos_tags if tag.startswith('JJ'))

    @staticmethod
    def count_named_entities(text):
        doc = nlp(text)
        return len([ent for ent in doc.ents])

# Load data and split into train/test
df = load_and_preprocess_data()
X = df[['review']]
y = df['label']

# Stratified split to handle imbalanced data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create a column transformer with both text vectorization and custom features
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=5), 'review'),
        ('custom_features', AdvancedFeatureTransformer(), ['review'])
    ]
)

# Pipeline with feature extraction and model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', C=1.0, random_state=42, class_weight='balanced'))
])

# Train the model
pipeline.fit(X_train, y_train)

# Cross-validation scores
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"Average CV Accuracy: {np.mean(scores):.2f}")

# Save the trained model
joblib.dump(pipeline, 'fake_review_detector.pkl')

# Prediction function
def predict_review(review):
    pipeline = joblib.load('fake_review_detector.pkl')  # Load model
    X_input = pd.DataFrame({'review': [preprocess_text(review)]})
    prediction = pipeline.predict(X_input)[0]
    confidence = pipeline.predict_proba(X_input)[0].max() * 100
    result = 'True' if prediction == 1 else 'Fake'
    return result, f"{confidence:.2f}%"

# Function to update the model with new data
def update_model(new_review, new_label):
    global X_train, y_train

    # Preprocess and append new data to training set
    new_review_processed = preprocess_text(new_review)
    X_train = pd.concat([X_train, pd.DataFrame({'review': [new_review_processed]})], ignore_index=True)
    y_train = pd.concat([y_train, pd.Series([new_label])], ignore_index=True)

    # Retrain the model with updated data
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, 'fake_review_detector.pkl')  # Save updated model

print("Model training complete and saved as 'fake_review_detector.pkl'.")
