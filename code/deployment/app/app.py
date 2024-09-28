# code/deployment/app/app.py
import streamlit as st
import requests

st.title("ML Prediction")

features = st.text_input("Enter features (comma-separated)")
if st.button("Predict"):
    if features:
        features_list = [float(x) for x in features.split(",")]
        response = requests.post("http://api:3500/predict/", json={"features": features_list})
        st.write(f"Prediction: {response.json()['prediction']}")
    else:
        st.write("Please enter the features")
