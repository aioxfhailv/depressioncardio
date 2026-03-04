import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Depression AI", layout="centered")

st.markdown("## Depression Severity Prediction")
st.markdown("Based on Beck Depression Inventory + physiological data")

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
    prediction = model.predict(data)[0]

    labels = {
        0: "No depression",
        1: "Mild depression",
        2: "Moderate depression",
        3: "Severe depression"
    }

    st.success(f"Predicted severity: {labels[prediction]}")

    proba = model.predict_proba(data)[0]

    st.markdown("### Probability by class:")
    for i, p in enumerate(proba):
        st.write(f"{labels[i]}: {round(p*100, 2)}%")
