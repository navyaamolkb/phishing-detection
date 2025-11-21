
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('model/phishing_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']
    X = vectorizer.transform([url])
    pred = model.predict(X)[0]
    return render_template('result.html', prediction=pred)

if __name__ == "__main__":
    app.run(debug=True)
