from flask import Flask, request, jsonify, render_template
from fake_review_detector import model_pipeline, preprocess_text

app = Flask(__name__)

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

    # Preprocess and predict
    cleaned_review = preprocess_text(review)
    prediction = model_pipeline.predict([cleaned_review])[0]
    confidence = max(model_pipeline.predict_proba([cleaned_review])[0]) * 100

    result = "Fake" if prediction == 1 else "Genuine"
    return jsonify({
        'prediction': result,
        'confidence': round(confidence, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
