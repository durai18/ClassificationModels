# Wine Quality Prediction Project

## a. Problem Statement
This project aims to perform a comparative analysis of multiple classification models to predict the quality of red wine based on its physicochemical properties. The goal is to identify the most effective model for this task by evaluating their performance across various metrics.

## b. Dataset Description
The project utilizes the 'Wine Quality (Red)' dataset from the UCI Machine Learning Repository. This dataset contains 1599 instances and 12 features, describing various physicochemical properties of red variants of Portuguese 'Vinho Verde' wine. The target variable is 'quality', an ordinal variable ranging from 3 (very poor) to 8 (very excellent), representing sensory data assessed by human experts.

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Features**: 11 physicochemical properties (e.g., fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol).
- **Target**: Wine quality (score between 3 and 8).

## c. Models Used and Performance Comparison

The following classification models were implemented and evaluated on the preprocessed dataset:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Random Forest Classifier (Ensemble Model)
6. XGBoost Classifier (Ensemble Model)

Each model was evaluated using Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC Score). The comparative performance is summarized in the table below:

### Model Performance Summary
|                      | Accuracy   | AUC Score   | Precision   | Recall   | F1 Score   | MCC Score   |
|:---------------------|:-----------|:------------|:------------|:---------|:-----------|:------------|
| Logistic Regression  | 0.5656     | 0.7478      | 0.5029      | 0.5656   | 0.5229     | 0.2848      |
| Decision Tree        | 0.5625     | 0.659       | 0.5533      | 0.5625   | 0.5576     | 0.3174      |
| K-Nearest Neighbors  | 0.5469     | 0.7198      | 0.5224      | 0.5469   | 0.5309     | 0.2694      |
| Gaussian Naive Bayes | 0.5469     | 0.7315      | 0.5426      | 0.5469   | 0.5435     | 0.302       |
| Random Forest        | 0.6594     | 0.8458      | 0.6309      | 0.6594   | 0.6438     | 0.4561      |
| XGBoost              | 0.7062     | 0.8526      | 0.6759      | 0.7062   | 0.6901     | 0.533       |

### Qualitative Observations on Model Performance
| Model Name               | Observation about Model Performance                                                               |
|:-------------------------|:--------------------------------------------------------------------------------------------------|
| Logistic Regression      | Moderate performance across metrics; good balance but not top tier.                               |
| Decision Tree            | Similar to Logistic Regression in accuracy, but lower AUC, suggesting weaker probability ranking. |
| K-Nearest Neighbors      | Lower accuracy and F1, indicating less effective classification.                                  |
| Gaussian Naive Bayes     | Moderate performance, slightly better precision than recall.                                      |
| Random Forest (Ensemble) | Strong improvement over single models, good accuracy and AUC.                                     |
| XGBoost (Ensemble)       | Best overall performance, highest accuracy, AUC, and MCC. 

## Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository (if applicable) or download the project files.**

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ensure models and scaler are available:**
    The trained models and the StandardScaler should be saved in a directory named `model/` in the project root.
    If you are running the notebook, these files will be generated automatically by executing the relevant cells.

## Running the Streamlit Application

Once the setup is complete, you can run the Streamlit application:

```bash
streamlit run app.py
```

This command will open the application in your web browser, where you can input wine properties and get quality predictions.


**General Deployment Steps:**

1.  **Host your project files** (including `app.py`, `requirements.txt`, and the `model/` directory) on a platform like GitHub.
2.  **Connect your repository** to a deployment service (e.g., Streamlit Cloud).
3.  **Specify `app.py`** as the main application file.
4.  The service will automatically install dependencies from `requirements.txt` and run your app.
