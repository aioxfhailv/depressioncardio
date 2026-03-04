import streamlit as st
import joblib
import numpy as np

model = joblib.load("bdi_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Depression Prediction (BDI)")

inputs = []

for i in range(1, 22):
    val = st.slider(f"BDI item {i}", 0, 3, 0)
    inputs.append(val)

hr = st.number_input("Heart Rate", 50, 150)
sys = st.number_input("Systolic BP", 80, 200)
dia = st.number_input("Diastolic BP", 40, 130)

inputs.extend([hr, sys, dia])

if st.button("Predict"):
    data = scaler.transform([inputs])
    prediction = model.predict(data)
    st.success(f"Predicted severity: {prediction[0]}")
