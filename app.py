# streamlit_app.py
import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("heart_model.pkl")

st.title("❤️ Heart Disease Prediction App")

# Collect input features
age = st.number_input("Age", 0, 120)
sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
cp = st.number_input("Chest Pain Type (0-3)", 0, 3)
trestbps = st.number_input("Resting Blood Pressure", 0, 300)
chol = st.number_input("Serum Cholestoral (mg/dl)", 0, 600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [1, 0])
restecg = st.number_input("Resting ECG (0-2)", 0, 2)
thalach = st.number_input("Maximum Heart Rate Achieved", 0, 250)
exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [1, 0])
oldpeak = st.number_input("ST Depression", 0.0, 10.0, step=0.1)
slope = st.number_input("Slope of Peak Exercise ST Segment (0-2)", 0, 2)
ca = st.number_input("Number of Major Vessels (0-4)", 0, 4)
thal = st.number_input("Thal (1-3)", 1, 3)

if st.button("Predict"):
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(features)[0]

    if prediction == 1:
        st.error("⚠️ High risk of heart disease.")
    else:
        st.success("✅ Low risk of heart disease.")
