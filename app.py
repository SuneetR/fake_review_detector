import os
import pickle
import logging
from flask import Flask, request, jsonify, render_template

# Initialize Flask app
app = Flask(__name__)

# Set up logging (set to WARNING to reduce verbosity in production)
logging.basicConfig(level=logging.WARNING)

# Load model components
def load_model_components():
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        with open('pca.pkl', 'rb') as f:
            pca = pickle.load(f)
        with open('stacking_model.pkl', 'rb') as f:
            stacking_model = pickle.load(f)
        return tfidf_vectorizer, pca, stacking_model
    except Exception as e:
        logging.error(f"Error loading model components: {e}")
        raise RuntimeError("Model components could not be loaded.")

# Predict function
def predict_review(review, tfidf_vectorizer, pca, stacking_model):
    # Preprocess the review
    review = review.lower()
    tokens = review.split()
    tokens = [word.strip(string.punctuation) for word in tokens if word not in stop_words]
    cleaned_review = ' '.join(tokens)

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

# Load components on startup
tfidf_vectorizer, pca, stacking_model = load_model_components()

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
