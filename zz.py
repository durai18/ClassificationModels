import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Define the directory where models are saved
MODEL_DIR = 'model/'

# Load the best-performing model (XGBoost)
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

# Load the StandardScaler
@st.cache_resource
def load_scaler(scaler_path):
    return joblib.load(scaler_path)

xgb_model = load_model(os.path.join(MODEL_DIR, 'xgboost_model.joblib'))
scaler = load_scaler(os.path.join(MODEL_DIR, 'scaler_wine_quality.joblib'))

# Define the minimum quality level for re-indexing
MIN_QUALITY_LEVEL = 3 # Based on the dataset analysis (wine quality range 3-8)

# Define the feature names in the order they were used for training
# This order is crucial for correct predictions
feature_names = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]

st.title('Wine Quality Prediction App')
st.write('Enter the physicochemical properties of the wine to predict its quality.')

# Create input fields for each feature
st.sidebar.header('Wine Properties Input')

input_data = {}
for feature in feature_names:
    # Provide reasonable default values or ranges based on dataset description (if known) or general knowledge
    if feature == 'fixed acidity':
        input_data[feature] = st.sidebar.number_input(f'Enter {feature}:', min_value=4.0, max_value=16.0, value=7.4, step=0.1)
    elif feature == 'volatile acidity':
        input_data[feature] = st.sidebar.number_input(f'Enter {feature}:', min_value=0.1, max_value=1.6, value=0.70, step=0.01)
    elif feature == 'citric acid':
        input_data[feature] = st.sidebar.number_input(f'Enter {feature}:', min_value=0.0, max_value=1.0, value=0.00, step=0.01)
    elif feature == 'residual sugar':
        input_data[feature] = st.sidebar.number_input(f'Enter {feature}:', min_value=0.5, max_value=16.0, value=1.9, step=0.1)
    elif feature == 'chlorides':
        input_data[feature] = st.sidebar.number_input(f'Enter {feature}:', min_value=0.01, max_value=0.7, value=0.076, step=0.001)
    elif feature == 'free sulfur dioxide':
        input_data[feature] = st.sidebar.number_input(f'Enter {feature}:', min_value=1.0, max_value=72.0, value=11.0, step=1.0)
    elif feature == 'total sulfur dioxide':
        input_data[feature] = st.sidebar.number_input(f'Enter {feature}:', min_value=6.0, max_value=300.0, value=34.0, step=1.0)
    elif feature == 'density':
        input_data[feature] = st.sidebar.number_input(f'Enter {feature}:', min_value=0.990, max_value=1.005, value=0.9978, step=0.0001, format="%.4f")
    elif feature == 'pH':
        input_data[feature] = st.sidebar.number_input(f'Enter {feature}:', min_value=2.7, max_value=4.0, value=3.51, step=0.01)
    elif feature == 'sulphates':
        input_data[feature] = st.sidebar.number_input(f'Enter {feature}:', min_value=0.3, max_value=2.0, value=0.56, step=0.01)
    elif feature == 'alcohol':
        input_data[feature] = st.sidebar.number_input(f'Enter {feature}:', min_value=8.0, max_value=15.0, value=9.4, step=0.1)

# Display input data for verification
st.subheader('Entered Wine Properties')
input_df = pd.DataFrame([input_data])
st.write(input_df)

if st.button('Predict Wine Quality'):
    # Ensure the order of columns in the input DataFrame matches the training data
    input_df = input_df[feature_names]
    
    # Scale the input data
    scaled_input = scaler.transform(input_df)
    
    # Make prediction
    prediction_reindexed = xgb_model.predict(scaled_input)
    
    # Add back the min_quality_level to get the actual quality score
    predicted_quality = prediction_reindexed[0] + MIN_QUALITY_LEVEL
    
    st.subheader('Predicted Wine Quality')
    st.success(f'The predicted wine quality is: **{int(predicted_quality)}**')
