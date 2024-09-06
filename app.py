# Import necessary libraries
import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import f1_score, make_scorer
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from torch.nn.functional import softmax
from scipy.sparse import hstack
from nltk.corpus import stopwords
import random
from sklearn.model_selection import cross_val_score
from gensim.models import Word2Vec
from scipy.sparse import csr_matrix


# Download required NLTK resources
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the dataset
data = pd.read_csv('reviews.csv')  # Replace with your dataset path
data['label'] = data['label'].map({'fake': 0, 'genuine': 1})  # Adjust according to your dataset

# Data Augmentation: Synonym Replacement
def synonym_replacement(text):
    # Example synonym replacement using WordNet
    words = nltk.word_tokenize(text)
    new_text = []
    for word in words:
        synonyms = nltk.corpus.wordnet.synsets(word)
        if synonyms:
            synonym = random.choice(synonyms).lemmas()[0].name()
            new_text.append(synonym if synonym != word else word)
        else:
            new_text.append(word)
    return ' '.join(new_text)

# Data Augmentation: Random Deletion
def random_deletion(text, p=0.1):
    # Deletes random words with probability p
    words = nltk.word_tokenize(text)
    if len(words) == 1: 
        return text
    new_text = [word for word in words if random.random() > p]
    return ' '.join(new_text)

# Data Augmentation: Apply augmentations
data_augmented = data.copy()
data_augmented['cleaned_review'] = data['review'].apply(preprocess_text)
data_augmented['cleaned_review'] = data_augmented['cleaned_review'].apply(lambda x: synonym_replacement(x))
data_augmented = pd.concat([data, data_augmented], ignore_index=True)

# Preprocessing function
def preprocess_text(text):
    # Tokenize text
    tokens = word_tokenize(text.lower())
    # Remove punctuation and stopwords
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return ' '.join(cleaned_tokens)

# Load data
data = pd.read_csv('reviews.csv')

# Apply preprocessing
data['cleaned_review'] = data['review'].apply(preprocess_text)

# Combine original and augmented data
data = pd.concat([data, data_augmented], ignore_index=True)

# Split the dataset
X = data['cleaned_review']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Transformer embeddings using DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def get_transformer_embeddings(text_data):
    inputs = tokenizer(text_data.tolist(), padding=True, truncation=True, return_tensors="pt", max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Generate embeddings
X_train_bert = get_transformer_embeddings(X_train)
X_test_bert = get_transformer_embeddings(X_test)

# Combine TF-IDF and BERT embeddings
X_train_combined = hstack([X_train_tfidf, X_train_bert])
X_test_combined = hstack([X_test_tfidf, X_test_bert])

# Define base models
lr = LogisticRegression(max_iter=2000, class_weight='balanced', C=0.1)
nb = MultinomialNB(alpha=0.5)
svc = SVC(kernel='linear', probability=True, class_weight='balanced', C=1)

# Ensemble model with voting
ensemble_model = VotingClassifier(estimators=[
    ('lr', lr),
    ('nb', nb),
    ('svc', svc)
], voting='soft')

# Cross-validation setup
cv = StratifiedKFold(n_splits=5)

# Hyperparameter tuning
param_grid = {
    'lr__C': [0.01, 0.1, 1, 10],
    'svc__C': [0.01, 0.1, 1, 10],
    'nb__alpha': [0.1, 0.5, 1]
}

grid_search = GridSearchCV(ensemble_model, param_grid, cv=cv, scoring=make_scorer(f1_score, average='weighted'))
grid_search.fit(X_train_combined, y_train)

# Fit the model on the best parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train_combined, y_train)

# Predict and evaluate
y_pred = best_model.predict(X_test_combined)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Optimized F1 score: {f1:.4f}")

# Save the trained model
import joblib
joblib.dump(best_model, 'optimized_fake_review_detector_model.pkl')
