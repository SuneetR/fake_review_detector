import os
from flask import Flask, request, jsonify, render_template
from fake_review_detector import predict_review, load_reviews, train_model
import logging
import gc
import pickle

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

MODEL_PATH = 'trained_model.pkl'

def load_or_train_model():
    """
    Load the model from disk if it exists, otherwise train a new one and save it.
    """
    if os.path.exists(MODEL_PATH):
        logging.debug(f"Loading pre-trained model from {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as model_file:
            return pickle.load(model_file)
    else:
        logging.debug("No pre-trained model found. Training the model from scratch...")
        df = load_reviews('reviews.csv')
        reviews = df['review']
        labels = df['label']
        model = train_model(reviews, labels)
        logging.debug(f"Saving the trained model to {MODEL_PATH}")
        with open(MODEL_PATH, 'wb') as model_file:
            pickle.dump(model, model_file)
        return model

# Load or train the model when the app starts
logging.debug("Initializing app and loading or training model...")
model = load_or_train_model()

@app.route('/')
def home():
    logging.debug("Rendering homepage.")
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        review = data.get('review')
        logging.debug(f"Review received: {review}")

        if not review:
            logging.error("No review provided.")
            return jsonify({'error': 'No review provided.'}), 400

        prediction, confidence = predict_review(review)
        logging.debug(f"Prediction: {prediction}, Confidence: {confidence}")
        gc.collect()

        return jsonify({
            'prediction': prediction,
            'confidence': confidence
        })
    except Exception as e:
        logging.error(f"Error analyzing review: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use the PORT environment variable provided by Render
    port = int(os.environ.get("PORT", 5000))  # Default to 5000
    app.run(host='0.0.0.0', port=port, debug=True)
