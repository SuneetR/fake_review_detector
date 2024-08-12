from flask import Flask, request, render_template
from fake_review_detector import predict_review

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    result, confidence = predict_review(review)
    return render_template('index.html', prediction=result, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
