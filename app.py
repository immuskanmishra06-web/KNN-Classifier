import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- 1. Load the Saved Model and Scaler ---
@st.cache_resource # This caches the models so they don't reload on every button click
def load_components():
    with open('knn_model.pkl', 'rb') as model_file:
        knn = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return knn, scaler

knn_model, scaler = load_components()

# --- 2. Build the Streamlit UI ---
st.title("🎓 Student Performance Predictor")
st.write("Enter the student's details below to predict if they will pass.")

# Create input fields for the user
col1, col2 = st.columns(2)

with col1:
    study_hours = st.number_input("Study Hours per Week", min_value=0.0, max_value=168.0, value=10.0, step=0.5)
    attendance = st.number_input("Attendance Rate (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
    grades = st.number_input("Previous Grades (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)

with col2:
    # Dropdowns for categorical data
    extracurricular = st.selectbox("Participation in Extracurricular Activities", ["Yes", "No"])
    parent_education = st.selectbox("Parent Education Level", ["High School", "Associate", "Bachelor", "Master"])

# --- 3. Process Inputs and Predict ---
if st.button("Predict Performance"):
    
    # Map categorical inputs to the numbers LabelEncoder used (Alphabetical order)
    # Extracurricular: No = 0, Yes = 1
    extra_encoded = 1 if extracurricular == "Yes" else 0
    
    # Parent Education: Associate = 0, Bachelor = 1, High School = 2, Master = 3
    education_mapping = {"Associate": 0, "Bachelor": 1, "High School": 2, "Master": 3}
    parent_encoded = education_mapping[parent_education]
    
    # Create a DataFrame for the input matching the training columns EXACTLY
    input_data = pd.DataFrame([[
        study_hours, 
        attendance, 
        grades, 
        extra_encoded, 
        parent_encoded
    ]], columns=[
        'Study Hours per Week', 
        'Attendance Rate', 
        'Previous Grades', 
        'Participation in Extracurricular Activities', 
        'Parent Education Level'
    ])
    
    # Scale the numerical features using the loaded scaler
    scaled_input = scaler.transform(input_data.values)
    
    # Make the prediction
    prediction = knn_model.predict(scaled_input)
    
    # LabelEncoder makes 'No' = 0 and 'Yes' = 1 for the 'Passed' column
    if prediction[0] == 1:
        st.success("✅ Prediction: This student is likely to PASS!")
        st.balloons()
    else:
        st.error("❌ Prediction: This student is at risk of FAILING.")
