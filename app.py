from flask import Flask, request, jsonify, render_template
from fake_review_detector import predict_review, load_reviews, train_model
import logging
import gc

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Train the model when the app starts
logging.debug("Loading reviews.csv and training the model...")
df = load_reviews('reviews.csv')
reviews = df['review']
labels = df['label']
train_model(reviews, labels)

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
    app.run(debug=True)
