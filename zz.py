import streamlit as st
import joblib # Changed from pickle to joblib
import pandas as pd
import os # Added os for path joining
import numpy as np # Added numpy for consistency with data types

st.title("Wine Quality Classification")

# Define the directory where models are saved
MODEL_DIR = 'model/'

# Load trained model (XGBoost is the best performing) and scaler
try:
    # Load the XGBoost model saved with joblib
    model = joblib.load(os.path.join(MODEL_DIR, 'xgboost_model.joblib'))
    # Load the StandardScaler saved with joblib
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler_wine_quality.joblib'))
    # The minimum quality level is 3, used for re-indexing
    min_quality_level = 3
    st.success("Model and Scaler loaded successfully.")
except FileNotFoundError as e:
    st.error(f"Error loading model or scaler: {e}. Make sure the 'model/' directory exists and contains 'xgboost_model.joblib' and 'scaler_wine_quality.joblib'.")
    st.stop() # Stop Streamlit app if model cannot be loaded
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.stop()

# Define the feature names in the order they were used for training
# This order is crucial for correct predictions
feature_names = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]

st.sidebar.header('Wine Properties Input')

input_data = {}
for feature in feature_names:
    # Provide reasonable default values and ranges based on dataset description
    if feature == 'fixed acidity':
        input_data[feature] = st.sidebar.number_input(f'Enter {feature}:', min_value=4.6, max_value=15.9, value=7.4, step=0.1)
    elif feature == 'volatile acidity':
        input_data[feature] = st.sidebar.number_input(f'Enter {feature}:', min_value=0.12, max_value=1.58, value=0.70, step=0.01)
    elif feature == 'citric acid':
        input_data[feature] = st.sidebar.number_input(f'Enter {feature}:', min_value=0.0, max_value=1.0, value=0.00, step=0.01)
    elif feature == 'residual sugar':
        input_data[feature] = st.sidebar.number_input(f'Enter {feature}:', min_value=0.9, max_value=15.5, value=1.9, step=0.1)
    elif feature == 'chlorides':
        input_data[feature] = st.sidebar.number_input(f'Enter {feature}:', min_value=0.012, max_value=0.611, value=0.076, step=0.001, format="%.3f")
    elif feature == 'free sulfur dioxide':
        input_data[feature] = st.sidebar.number_input(f'Enter {feature}:', min_value=1.0, max_value=72.0, value=11.0, step=1.0)
    elif feature == 'total sulfur dioxide':
        input_data[feature] = st.sidebar.number_input(f'Enter {feature}:', min_value=6.0, max_value=289.0, value=34.0, step=1.0)
    elif feature == 'density':
        input_data[feature] = st.sidebar.number_input(f'Enter {feature}:', min_value=0.9900, max_value=1.0037, value=0.9978, step=0.0001, format="%.4f")
    elif feature == 'pH':
        input_data[feature] = st.sidebar.number_input(f'Enter {feature}:', min_value=2.74, max_value=4.01, value=3.51, step=0.01)
    elif feature == 'sulphates':
        input_data[feature] = st.sidebar.number_input(f'Enter {feature}:', min_value=0.33, max_value=2.0, value=0.56, step=0.01)
    elif feature == 'alcohol':
        input_data[feature] = st.sidebar.number_input(f'Enter {feature}:', min_value=8.4, max_value=14.9, value=9.4, step=0.1)

st.subheader('Entered Wine Properties')
input_df = pd.DataFrame([input_data])
st.write(input_df)

if st.button("Predict Quality"):
    # Ensure the order of columns in the input DataFrame matches the training data
    input_df = input_df[feature_names]
    
    # Scale the input data using the loaded scaler
    scaled_input = scaler.transform(input_df)
    
    # Make prediction (output will be 0-indexed)
    prediction_reindexed = model.predict(scaled_input)
    
    # Add back the min_quality_level to get the actual quality score
    predicted_quality = prediction_reindexed[0] + min_quality_level
    
    st.success(f"Predicted Wine Quality: {int(predicted_quality)}")
