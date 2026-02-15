import streamlit as st
import pickle
import pandas as pd

st.title("Wine Quality Classification")

# Load trained model
with open("model/trained_model.pkl", "rb") as f:
    model = pickle.load(f)

# User input
fixed_acidity = st.number_input("Fixed Acidity", 5.0, 15.0, 7.4)
volatile_acidity = st.number_input("Volatile Acidity", 0.1, 1.5, 0.7)
citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.0)
residual_sugar = st.number_input("Residual Sugar", 0.5, 15.0, 1.9)
alcohol = st.number_input("Alcohol", 8.0, 15.0, 9.4)

if st.button("Predict Quality"):
    features = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, alcohol]],
                            columns=["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "alcohol"])
    prediction = model.predict(features)
    st.success(f"Predicted Wine Quality: {prediction[0]}")
