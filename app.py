import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('linear_regression_model.pkl')

# Set up the Streamlit interface
st.title("Sales Prediction App")

# Create input fields
tv = st.number_input("TV Advertising Budget (in thousands)", min_value=0.0, step=0.1)
radio = st.number_input("Radio Advertising Budget (in thousands)", min_value=0.0, step=0.1)
newspaper = st.number_input("Newspaper Advertising Budget (in thousands)", min_value=0.0, step=0.1)

# Prediction button
if st.button("Predict Sales"):
    # Create the feature array
    features = [[tv, radio, newspaper]]

    # Make a prediction
    prediction = model.predict(features)[0]

    # Display the result
    st.success(f"Predicted Sales: {round(prediction, 2)} units")
