# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 20:39:48 2025

@author: HP
"""
import numpy as np
import pickle
import streamlit as st
import os

# Load the model
model_path = "C:\\Users\\HP\\Downloads\\Diabetesprediction\\trained_model.sav"

if os.path.exists(model_path):
    try:
        loaded_model = pickle.load(open(model_path, "rb"))
        print("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"⚠ Error loading model: {e}")
else:
    st.error(f"⚠ Model file not found at {model_path}")

# Load the scaler if it exists
scaler_path = "C:\\Users\\HP\\Downloads\\Diabetesprediction\\scaler.sav"
scaler = None
if os.path.exists(scaler_path):
    try:
        scaler = pickle.load(open(scaler_path, "rb"))
        print("✅ Scaler loaded successfully!")
    except Exception as e:
        st.error(f"⚠ Error loading scaler: {e}")

# Function for prediction
def diabetes_prediction(input_data):
    try:
        # Convert input data to numpy array with correct data type
        input_data_as_array = np.asarray(input_data, dtype=np.float64).reshape(1, -1)

        # Apply scaling if scaler is available
        if scaler:
            input_data_as_array = scaler.transform(input_data_as_array)

        # Predict output
        prediction = loaded_model.predict(input_data_as_array)
        print(f"🔍 Raw Model Output: {prediction}")  # Debugging output

        return "The person is Not Diabetic" if prediction[0] == 0 else "The person is Diabetic"

    except Exception as e:
        return f"⚠ Error in prediction: {e}"

# Streamlit App
def main():
    st.title("Diabetes Prediction Web Application")

    # Get input from users
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure Value")
    SkinThickness = st.text_input("Skin Thickness")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI Value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
    Age = st.text_input("Age of Person")

    # Prediction Output
    diagnosis = ""

    # Prediction button
    if st.button("Diabetes Test Results"):
        try:
            # Convert inputs to float before passing to model
            user_input = [
                float(Pregnancies),
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                float(Age),
            ]
            diagnosis = diabetes_prediction(user_input)

        except ValueError:
            diagnosis = "⚠ Please enter valid numeric values."

    st.success(diagnosis)

if __name__ == "__main__":
    main()
