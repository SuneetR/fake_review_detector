import os
import logging
import string
from flask import Flask, request, jsonify, render_template
from fake_review_detector import preprocess_text, train_model, load_data

# Initialize Flask app
app = Flask(__name__)

# Set up logging (set to WARNING to reduce verbosity in production)
logging.basicConfig(level=logging.WARNING)

# Load and train the model on startup
X, y = load_data()  # Load data from the CSV file
tfidf_vectorizer, pca, stacking_model = train_model(X, y)  # Train the model

# Predict function
def predict_review(review, tfidf_vectorizer, pca, stacking_model):
    # Preprocess the review
    cleaned_review = preprocess_text(review)

    # Vectorize and apply PCA transformation
    tfidf_features = tfidf_vectorizer.transform([cleaned_review])
    tfidf_features_pca = pca.transform(tfidf_features.toarray())

    # Predict using the model
    probabilities = stacking_model.predict_proba(tfidf_features_pca)[0]
    prob_genuine = probabilities[0]
    prob_fake = probabilities[1]

    prediction = "Fake" if prob_fake > prob_genuine else "Genuine"
    confidence = round(max(prob_genuine, prob_fake) * 100, 2)

    return prediction, confidence

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# API route for prediction
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    review = data.get('review', '')

    if not review:
        return jsonify({'error': 'No review provided'}), 400

    # Get prediction
    prediction, confidence = predict_review(review, tfidf_vectorizer, pca, stacking_model)

    return jsonify({
        'prediction': prediction,
        'confidence': confidence
    })

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
