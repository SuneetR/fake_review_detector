from flask import Flask, request, jsonify, render_template
from fake_review_detector import predict_review  # Import the predict function
import nltk
import gc

# Ensure NLTK resources are available
nltk.download('stopwords')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML template with the form

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()  # Get the JSON data from the request
    review = data.get('review')  # Extract the review text
    if not review:
        return jsonify({'error': 'No review provided.'}), 400
    
    try:
        # Call the AI model to predict if the review is fake or genuine
        prediction, confidence = predict_review(review)
        gc.collect()  # Clean up memory
        return jsonify({
            'prediction': prediction,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
