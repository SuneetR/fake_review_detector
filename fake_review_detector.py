import string
import nltk
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# Load data from CSV file
df = pd.read_csv('reviews.csv')

# Data preprocessing function
def preprocess_text(text):
    text = text.lower()  # Lowercase text
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Apply preprocessing to the dataset
df['review'] = df['review'].apply(preprocess_text)

# Split the dataset into training and test sets
X = df['review']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with TfidfVectorizer and LogisticRegression
pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression())

# Train the model
pipeline.fit(X_train, y_train)

# Function to predict if a review is fake or genuine
def predict_review(review):
    # Preprocess the review
    preprocessed_review = preprocess_text(review)
    # Predict
    prediction = pipeline.predict([preprocessed_review])[0]
    confidence = pipeline.predict_proba([preprocessed_review])[0].max()
    # Interpret the result
    result = 'Fake' if prediction == 1 else 'Genuine'
    return result, confidence
