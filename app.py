import streamlit as st
import pandas as pd
import pickle

# Load the trained model
try:
    model = pickle.load(open('revenue_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: revenue_model.pkl not found. Please ensure the model file is in the project folder.")
    st.stop()

# Streamlit app
st.title("YouTube Revenue Predictor")

# Input fields with default values
duration = st.number_input("Video Duration (seconds)", min_value=0, value=300, step=1)
views = st.number_input("Estimated Views", min_value=0, value=10000, step=100)
likes = st.number_input("Estimated Likes", min_value=0, value=500, step=10)
shares = st.number_input("Estimated Shares", min_value=0, value=50, step=5)
subscribers = st.number_input("Estimated New Subscribers", min_value=0, value=10, step=1)

# Prediction button
if st.button("Predict Revenue"):
    # Create input DataFrame with correct feature names
    input_data = pd.DataFrame([[duration, views, likes, shares, subscribers]], 
                              columns=['Video Duration', 'Views', 'Likes', 'Shares', 'New Subscribers'])
    try:
        prediction = model.predict(input_data)[0]
        st.write(f"Predicted Revenue: ${prediction:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")