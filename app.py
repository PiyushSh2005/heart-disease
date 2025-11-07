# app.py
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("heart_model.pkl")

# Home route (renders HTML form)
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(final_features)
        output = int(prediction[0])

        # Return result
        if output == 1:
            result = "⚠️ The model predicts a high risk of heart disease."
        else:
            result = "✅ The model predicts a low risk of heart disease."

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return jsonify({'error': str(e)})

# API endpoint (for programmatic access)
@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json(force=True)
        features = np.array(list(data.values())).reshape(1, -1)
        prediction = model.predict(features)
        output = int(prediction[0])
        return jsonify({'prediction': output})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
