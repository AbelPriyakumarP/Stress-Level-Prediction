import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Stress Level Predictor", page_icon="🧠", layout="centered")

# Title and description
st.title("🧠 Student Stress Level Predictor")
st.markdown("""
This app predicts a student's stress level (**Low**, **Moderate**, or **High**) based on their daily activities and GPA.
Enter the details below and click **Predict** to see the categorical stress level prediction and probabilities.
""")

# Load the saved Random Forest model, MinMaxScaler, and LabelEncoder
try:
    rf_model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("minmax_scaler.pkl")
    le = joblib.load("label_encoder.pkl")
except FileNotFoundError:
    st.error("Model, scaler, or label encoder file not found. Please ensure 'random_forest_model.pkl', 'minmax_scaler.pkl', and 'label_encoder.pkl' are in the same directory.")
    st.stop()

# Define feature names (based on the dataset)
features = [
    "Study_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day",
    "GPA"
]

# Define desired order for stress level categories
stress_levels_order = ["Low", "Moderate", "High"]

# Create input fields for each feature with validation
st.subheader("Enter Student Details")
input_data = {}
with st.form(key="input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        input_data["Study_Hours_Per_Day"] = st.number_input(
            "Study Hours Per Day", min_value=0.0, max_value=24.0, value=7.0, step=0.1
        )
        input_data["Extracurricular_Hours_Per_Day"] = st.number_input(
            "Extracurricular Hours Per Day", min_value=0.0, max_value=24.0, value=2.0, step=0.1
        )
        input_data["Sleep_Hours_Per_Day"] = st.number_input(
            "Sleep Hours Per Day", min_value=0.0, max_value=24.0, value=7.0, step=0.1
        )
    
    with col2:
        input_data["Social_Hours_Per_Day"] = st.number_input(
            "Social Hours Per Day", min_value=0.0, max_value=24.0, value=3.0, step=0.1
        )
        input_data["Physical_Activity_Hours_Per_Day"] = st.number_input(
            "Physical Activity Hours Per Day", min_value=0.0, max_value=12.9, value=4.0, step=0.1,
            help="Capped at 12.9 due to outlier in training data"
        )
        input_data["GPA"] = st.number_input(
            "GPA (0.0 to 4.0)", min_value=0.0, max_value=4.0, value=3.0, step=0.01,
            help="GPA capped at 4.0 based on standard academic scale"
        )
    
    # Submit button
    submit_button = st.form_submit_button(label="Predict Stress Level")

# Validate total hours
total_hours = (
    input_data["Study_Hours_Per_Day"] +
    input_data["Extracurricular_Hours_Per_Day"] +
    input_data["Sleep_Hours_Per_Day"] +
    input_data["Social_Hours_Per_Day"] +
    input_data["Physical_Activity_Hours_Per_Day"]
)
if submit_button and total_hours > 24:
    st.error(f"Total hours ({total_hours:.1f}) exceed 24 hours. Please adjust the input values.")
else:
    # Process the prediction when the form is submitted
    if submit_button:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data], columns=features)
        
        # Scale the input data using the loaded MinMaxScaler
        try:
            input_scaled = scaler.transform(input_df)
        except Exception as e:
            st.error(f"Error scaling input data: {e}")
            st.stop()
        
        # Make prediction
        try:
            prediction_encoded = rf_model.predict(input_scaled)[0]
            predicted_stress_level = le.inverse_transform([prediction_encoded])[0]
            st.success(f"Predicted Stress Level: **{predicted_stress_level}**")
            
            # Display prediction probabilities in specified order (Low, Moderate, High)
            st.subheader("Prediction Probabilities")
            probabilities = rf_model.predict_proba(input_scaled)[0]
            # Map model classes to probabilities
            class_prob_dict = dict(zip(le.inverse_transform(rf_model.classes_), probabilities))
            # Create ordered probabilities
            ordered_probs = [class_prob_dict.get(level, 0.0) for level in stress_levels_order]
            prob_df = pd.DataFrame({
                "Stress Level": stress_levels_order,
                "Probability": ordered_probs
            })
            st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}))

            # Display feature importance
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                "Feature": features,
                "Importance": rf_model.feature_importances_
            }).sort_values(by="Importance", ascending=False)
            
            # Plot feature importance
            fig, ax = plt.subplots(figsize=(8, 4))
            plt.bar(feature_importance["Feature"], feature_importance["Importance"])
            plt.xticks(rotation=45, ha="right")
            plt.xlabel("Feature")
            plt.ylabel("Importance")
            plt.title("Feature Importance in Random Forest Model")
            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.stop()

# Add footer
st.markdown("""
---
**Note**: This app uses a Random Forest Classifier to predict categorical stress levels (**Low**, **Moderate**, **High**).
Inputs should reflect realistic daily routines (total hours ≤ 24, GPA ≤ 4.0). The model was trained on student data with MinMaxScaler.
Key predictors include GPA, Study Hours, and Social Hours.
""")