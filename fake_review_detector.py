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

# Sample dataset
data = {
    'review': [
        'This product is great! I loved it and would buy again.',
        'Terrible product, it broke after one use.',
        'Amazing quality, really satisfied with the purchase.',
        'Do not buy this, it is a waste of money.',
        'Best purchase I ever made, highly recommended!',
        'Worst product ever, completely useless.',
        'Superb! Exceeded all my expectations.',
        'Not worth the price, very disappointed.',
        'I’m in love with this! Excellent quality.',
        'Completely dissatisfied, it stopped working in a week.',
        'Fantastic product, will recommend to everyone.',
        'Fake product, nothing like the description.',
        'Great value for the money, very happy.',
        'Total scam, don’t waste your money.',
        'This is the best thing I’ve bought this year.',
        'Horrible, I regret buying it.',
        'Top-notch, worth every penny.',
        'Broke within days, very poor quality.',
        'I’m very satisfied with my purchase.',
        'It’s a rip-off, don’t buy it.',
        'Absolutely love it, it’s perfect.',
        'Useless, complete waste of money.',
        'This product changed my life for the better.',
        'Didn’t work as advertised, very disappointed.',
        'Super happy with this, highly recommend!',
        'Trash, it’s not what they claim it to be.',
        'Excellent, will definitely buy again.',
        'Feels cheap, not worth the money.',
        'So happy with my purchase, amazing quality.',
        'Worst purchase ever, totally fake.',
        'Highly recommend this, great product!',
        'Don’t buy this, it’s a scam.',
        'The quality is top-notch, very satisfied.',
        'Terrible, it broke the first time I used it.',
        'This is exactly what I needed, very happy.',
        'Completely useless, waste of money.',
        'Best product in its category, very pleased.',
        'Fake and poorly made, don’t waste your money.',
        'I’m thrilled with this, great value!',
        'Didn’t work at all, very disappointed.',
        'Absolutely amazing, will purchase again.',
        'It’s a fake, don’t trust the reviews.',
        'This is the most reliable product I’ve ever bought.',
        'I hate it, not what I expected at all.',
        'Amazing quality, very happy with my purchase.',
        'It’s a rip-off, poor quality.',
        'Super impressed with this, highly recommend.',
        'Didn’t live up to the hype, very disappointed.',
        'This is a fantastic product, love it!',
        'Complete waste of money, fake product.',
        'I’m very satisfied, great quality.',
        'Broke after a few uses, not worth it.',
        'Best thing I’ve ever bought, very happy.',
        'Terrible product, don’t waste your money.',
        'Exceeded my expectations, very satisfied.',
        'Didn’t work as expected, very disappointed.',
        'Love it! Will buy again for sure.',
        'It’s a scam, nothing like the description.',
        'Very happy with this, great value for money.',
        'Didn’t meet my expectations, very disappointed.',
        'Fantastic, will definitely buy again.',
        'Complete trash, do not recommend.',
        'This product is a game-changer, love it!',
        'Useless, stopped working after a week.',
        'Very pleased with this purchase, highly recommend.',
        'Broke after one use, terrible quality.',
        'The quality is amazing, very satisfied.',
        'It’s a scam, don’t waste your money.',
        'Super happy with my purchase, excellent quality.',
        'Didn’t work at all, very disappointed.',
        'This is the best purchase I’ve made in a long time.',
        'Fake product, not worth it.',
        'Great product, very satisfied with the quality.',
        'It’s a rip-off, don’t buy it.',
        'Absolutely love this, highly recommend!',
        'Useless, not what they claim it to be.',
        'Exceeded my expectations, will buy again.',
        'Terrible, broke within a few days.',
        'This product is fantastic, very happy with it.',
        'Total scam, don’t waste your money.',
        'Love it, the quality is top-notch.',
        'It’s fake, completely disappointed.',
        'This is exactly what I was looking for, very satisfied.',
        'Broke after a week, not worth the money.',
        'Very happy with this, great quality.',
        'It’s a scam, not as described.',
        'Super impressed, will definitely recommend.',
        'Didn’t work as expected, very disappointed.',
        'Amazing product, exceeded my expectations.',
        'Complete waste of money, fake product.',
        'This is the best product I’ve ever bought, love it!',
        'Terrible quality, broke after a few uses.',
        'Very satisfied with my purchase, highly recommend.',
        'It’s a rip-off, don’t waste your money.',
        'Superb quality, very happy with this.',
        'Didn’t live up to the hype, very disappointed.',
        'Fantastic product, will definitely buy again.',
        'Useless, not what they claim it to be.',
        'Love this product, exceeded my expectations.',
        'Broke after one use, very disappointed.'
    ],
    'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
              0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
              0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
              0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
              0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}


df = pd.DataFrame(data)

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
