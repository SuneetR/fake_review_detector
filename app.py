from flask import Flask, request, render_template, redirect, url_for
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

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback = request.form['feedback']
    print(f"Received feedback: {feedback}")
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
