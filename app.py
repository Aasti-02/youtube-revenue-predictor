import streamlit as st
import pandas as pd
import pickle

# Load the trained model pipeline (includes scaler and model)
try:
    pipeline = pickle.load(open('revenue_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: revenue_model.pkl not found. Please ensure the model file is in the project folder.")
    st.stop()

# Streamlit app with improved layout
st.title("🎥 YouTube Revenue Predictor")
st.markdown("""
**Welcome to the YouTube Revenue Predictor!**  
Enter your video metrics below to estimate the revenue. This app uses a Random Forest model for accurate predictions.
""")

# Input fields with default values
st.header("📊 Video Metrics")
with st.form(key="input_form"):
    duration = st.number_input("Video Duration (seconds)", min_value=0, value=300, step=1)
    views = st.number_input("Estimated Views", min_value=0, value=10000, step=100)
    likes = st.number_input("Estimated Likes", min_value=0, value=500, step=10)
    shares = st.number_input("Estimated Shares", min_value=0, value=50, step=5)
    subscribers = st.number_input("Estimated New Subscribers", min_value=0, value=10, step=1)

    # Use columns to place the CTR input and its explanation side by side
    col1, col2 = st.columns([3, 2])  # Split the layout into two columns
    with col1:
        ctr = st.number_input("Video Thumbnail CTR (%)", min_value=0.0, value=5.0, step=0.1)
    with col2:
        st.markdown("""
        **What is Video Thumbnail CTR (%)?**  
        It’s the percentage of viewers who click on your video after seeing the thumbnail.  
        Example: If 100 people see your thumbnail and 5 click, the CTR is 5%.
        """)

    # Prediction button
    submit_button = st.form_submit_button(label="Predict Revenue")

# Make prediction when the button is clicked
if submit_button:
    try:
        # Create input DataFrame with interaction terms
        input_data = pd.DataFrame([[duration, views, likes, shares, subscribers, ctr]], 
                                  columns=['Video Duration', 'Views', 'Likes', 'Shares', 
                                           'New Subscribers', 'Video Thumbnail CTR (%)'])
        input_data['Views_Likes_Interaction'] = input_data['Views'] * input_data['Likes']
        input_data['Views_CTR_Interaction'] = input_data['Views'] * input_data['Video Thumbnail CTR (%)']

        # Ensure the order of features matches training
        feature_order = ['Video Duration', 'Views', 'Likes', 'Shares', 'New Subscribers', 
                         'Video Thumbnail CTR (%)', 'Views_Likes_Interaction', 'Views_CTR_Interaction']
        input_data = input_data[feature_order]

        # Make prediction using the pipeline
        prediction = pipeline.predict(input_data)[0]
        st.success(f"**Prediction Complete!** 🎉 Predicted Revenue: **${prediction:.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Additional info
st.markdown("""
---
**Note:** This model uses advanced features like interaction terms and a Random Forest algorithm for improved accuracy.  
For more details, check the [GitHub repository](https://github.com/your-username/youtube-revenue-predictor).
""")
