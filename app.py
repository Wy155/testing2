import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- Load Models ---
@st.cache_resource

def load_models():
    models = {
        "Gaussian Naive Bayes": load("gaussian_nb_model.joblib"),
        "Gradient Boosting": load("gradient_boosting_model.joblib"),
        "XGBoost Classifier": load("xgb_classifier_model.joblib"),
        "Random Forest": load("random_forest_model.joblib")
    }
    return models

models = load_models()

# --- Page Title ---
st.title("Credit Risk Prediction Dashboard")
st.markdown("Input your information below to predict credit risk.")

# --- User Input Fields ---
st.subheader("Enter Applicant Details")

person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
person_income = st.number_input("Annual Income ($)", min_value=0, value=50000)
person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
person_emp_length = st.number_input("Employment Length (years)", min_value=0, value=5)
loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_amnt = st.number_input("Loan Amount ($)", min_value=0, value=10000)
loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=100.0, value=12.5)
loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, max_value=1.0, value=0.2)
cb_person_default_on_file = st.selectbox("Previously Defaulted", ["Y", "N"])
cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=5)

# --- Prepare input ---
input_dict = {
    "person_age": person_age,
    "person_income": person_income,
    "person_home_ownership": 0 if person_home_ownership == "RENT" else (1 if person_home_ownership == "OWN" else (2 if person_home_ownership == "MORTGAGE" else 3)),
    "person_emp_length": person_emp_length,
    "loan_intent": 0 if loan_intent == "EDUCATION" else (1 if loan_intent == "MEDICAL" else (2 if loan_intent == "VENTURE" else (3 if loan_intent == "PERSONAL" else (4 if loan_intent == "HOMEIMPROVEMENT" else 5)))),
    "loan_amnt": loan_amnt,
    "loan_int_rate": loan_int_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_default_on_file": 1 if cb_person_default_on_file == "Y" else 0,
    "cb_person_cred_hist_length": cb_person_cred_hist_length
}

input_data = np.array([list(input_dict.values())])

# --- Model Selection ---
selected_model = st.selectbox("Choose a model for prediction", list(models.keys()))

if st.button("Predict"):
    model = models[selected_model]

    try:
        y_pred = model.predict(input_data)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(input_data)[:, 1]
        else:
            y_prob = np.zeros_like(y_pred)

        # --- Display Prediction ---
        st.subheader("Prediction Result")
        st.write(f"Predicted Class: {int(y_pred[0])}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Footer
st.markdown("---")
st.caption("Developed with Streamlit 🚀")
