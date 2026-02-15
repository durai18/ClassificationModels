import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report

# Define the directory where models are saved
MODEL_DIR = 'model/'

# Define the models and their filenames
MODELS = {
    'Logistic Regression': 'log_reg.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'K-Nearest Neighbors': 'knn.pkl',
    'Gaussian Naive Bayes': 'naive_bayes.pkl',
    'Random Forest': 'random_forest.pkl',
    'XGBoost': 'xgboost.pkl'
}

# Load the StandardScaler
@st.cache_resource
def load_scaler(scaler_path):
    return joblib.load(scaler_path)

# Load the selected model dynamically
@st.cache_resource
def load_model(model_name):
    model_filename = MODELS[model_name]
    model_path = MODEL_DIR + model_filename # No os.path.join needed if MODEL_DIR has trailing slash
    return joblib.load(model_path)

# Load the metrics DataFrame
@st.cache_resource
def load_metrics_df(metrics_path):
    return joblib.load(metrics_path)

scaler = load_scaler(MODEL_DIR + 'scaler_wine_quality.joblib')
metrics_df = load_metrics_df(MODEL_DIR + 'metrics_df.joblib')

# Define the minimum quality level for re-indexing
MIN_QUALITY_LEVEL = 3 # Based on the dataset analysis (wine quality range 3-8)

st.title('Wine Quality Prediction App')
st.write('Enter the physicochemical properties of the wine to predict its quality or upload a CSV file.')

# Sidebar for Model Selection and Input
st.sidebar.header('Model Selection')
selected_model_name = st.sidebar.selectbox('Select a Classification Model', list(MODELS.keys()))
model = load_model(selected_model_name)
st.subheader(f'Selected Model: {selected_model_name}')

# Display evaluation metrics for the selected model
st.sidebar.subheader('Model Evaluation Metrics')
if selected_model_name in metrics_df.index:
    st.sidebar.write(metrics_df.loc[selected_model_name].to_markdown())
else:
    st.sidebar.write("Metrics not available for this model.")

feature_names = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]

# Option to upload CSV for predictions
st.header('Predict with Custom Data')
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
        st.subheader('Uploaded Data (First 5 rows)')
        st.write(input_df.head())

        # Ensure the order of columns in the input DataFrame matches the training data
        input_df_processed = input_df[feature_names]

        # Scale the input data
        scaled_input = scaler.transform(input_df_processed)

        # Make prediction
        prediction_reindexed = model.predict(scaled_input)
        predicted_quality = prediction_reindexed + MIN_QUALITY_LEVEL

        st.subheader('Predicted Wine Quality for Uploaded Data')
        prediction_df = pd.DataFrame({'Predicted Quality': predicted_quality})
        st.write(prediction_df)

        # Option to display classification report if true labels are present in uploaded CSV
        if 'quality' in input_df.columns:
            true_labels = input_df['quality']
            # Re-index true labels if the model predicts 0-based labels
            if selected_model_name == 'XGBoost': # XGBoost was trained on re-indexed labels
                true_labels_reindexed = true_labels - MIN_QUALITY_LEVEL
            else:
                true_labels_reindexed = true_labels # Other models trained on original labels
            
            # Filter out true labels that are not present in the model's target range if necessary.
            # For simplicity, we assume the input quality labels are within the original training range.
            
            st.subheader('Classification Report for Uploaded Data')
            # Need to ensure unique classes for classification report match what the model can predict
            # Or map predicted values back to original quality range for true labels.
            # For now, let's assume direct comparison. Handle errors if classes don't match.
            try:
                report = classification_report(true_labels_reindexed, prediction_reindexed, zero_division=0, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.write(report_df.round(2))
            except ValueError as e:
                st.warning(f"Could not generate classification report: {e}. Ensure target labels match model output.")


    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")

else: # Manual input section
    st.header('Predict with Manual Input')
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
    input_df_manual = pd.DataFrame([input_data])
    st.write(input_df_manual)

    if st.button('Predict Wine Quality (Manual)'):
        # Ensure the order of columns in the input DataFrame matches the training data
        input_df_processed = input_df_manual[feature_names]

        # Scale the input data
        scaled_input = scaler.transform(input_df_processed)

        # Make prediction using the selected model
        prediction_reindexed = model.predict(scaled_input)

        # Add back the min_quality_level to get the actual quality score
        predicted_quality = prediction_reindexed[0] + MIN_QUALITY_LEVEL

        st.subheader('Predicted Wine Quality')
        st.success(f'The predicted wine quality is: **{int(predicted_quality)}**')
