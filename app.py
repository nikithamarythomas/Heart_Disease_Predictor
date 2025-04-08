import streamlit as st
import pandas as pd
import joblib
from preprocessing import preprocess_data  # Make sure this exists and works properly

# Load models
models = {
    'Logistic Regression': joblib.load('Logistic Regression_model.pkl'),
    'Random Forest': joblib.load('Random Forest_model.pkl'),
    'Gradient Boosting': joblib.load('Gradient boosting_model.pkl'),
    'Neural Network': joblib.load('Neural Network_model.pkl')
}

# Streamlit App Header
st.title("Heart Disease Prediction App")
st.markdown("""
### Predict the likelihood of heart disease based on patient data.
Fill in the details below, select a model, and get the prediction instantly!
""")

# Sidebar: Input Features
st.sidebar.header("Patient Data Input")

age = st.sidebar.slider("Age", 18, 100, 50)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
chest_pain = st.sidebar.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "No Chest Pain"])
resting_bp = st.sidebar.number_input("Resting Blood Pressure (mmHg)", 80, 200, 120)
cholesterol = st.sidebar.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", ["Yes", "No"])
resting_ecg = st.sidebar.selectbox("Resting ECG Results", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
max_hr = st.sidebar.slider("Max Heart Rate Achieved", 50, 220, 150)
exercise_angina = st.sidebar.selectbox("Exercise-Induced Angina", ["Yes", "No"])
oldpeak = st.sidebar.number_input("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)
st_slope = st.sidebar.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

# Mappings to match training dataset values
sex_map = {"Male": "M", "Female": "F"}
chest_pain_map = {
    "Typical Angina": "TA",       # Check your training data; sometimes "Typical Angina" was labeled "TA"
    "Atypical Angina": "ATA",
    "Non-Anginal Pain": "NAP",
    "No Chest Pain": "ASY"
}
fasting_bs_map = {"Yes": 1, "No": 0}
exercise_angina_map = {"Yes": "Y", "No": "N"}
resting_ecg_map = {
    "Normal": "Normal",
    "ST-T Abnormality": "ST",
    "Left Ventricular Hypertrophy": "LVH"
}
st_slope_map = {
    "Upsloping": "Up",
    "Flat": "Flat",
    "Downsloping": "Down"
}

# Create DataFrame from user input
input_data = pd.DataFrame({
    'Age': [age],
    'Sex': [sex_map[sex]],
    'ChestPainType': [chest_pain_map[chest_pain]],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'FastingBS': [fasting_bs_map[fasting_bs]],
    'RestingECG': [resting_ecg_map[resting_ecg]],
    'MaxHR': [max_hr],
    'ExerciseAngina': [exercise_angina_map[exercise_angina]],
    'Oldpeak': [oldpeak],
    'ST_Slope': [st_slope_map[st_slope]]
})
#st.write("Input Data Columns:", input_data.columns.tolist())

# Preprocess the input using external preprocessing module
input_data_transformed = preprocess_data(input_data)

#  Model selection
selected_model = st.sidebar.selectbox("Select Model", list(models.keys()))
model = models[selected_model]

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data_transformed)[0]
    prediction_proba = model.predict_proba(input_data_transformed)[0][1]
    result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
    
    # Show Results
    st.success(f"Prediction: {result}")
    st.info(f"Confidence Score: {prediction_proba:.2f}")

