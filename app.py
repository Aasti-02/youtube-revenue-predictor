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

st.header("📊 Video Metrics")
with st.form(key="input_form"):
    duration = st.number_input("Video Duration (minutes)", min_value=0.0, value=5.0, step=0.1)
    views = st.number_input("Estimated Views", min_value=0, value=10000, step=100)
    likes = st.number_input("Estimated Likes", min_value=0, value=500, step=10)
    shares = st.number_input("Estimated Shares", min_value=0, value=50, step=5)
    subscribers = st.number_input("Estimated New Subscribers", min_value=0, value=10, step=1)

    # Add hover tooltip for Video Thumbnail CTR (%) using HTML and CSS
    st.markdown("""
    <style>
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 220px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -110px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    <div class="tooltip">Video Thumbnail CTR (%)
        <span class="tooltiptext">The percentage of viewers who click on your video after seeing the thumbnail. Example: If 100 people see your thumbnail and 5 click, the CTR is 5%.</span>
    </div>
    """, unsafe_allow_html=True)
    ctr = st.number_input("", min_value=0.0, value=5.0, step=0.1, key="ctr")

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
        prediction = pipeline.predict(input_data)[0]*10
        st.success(f"**Prediction Complete!** 🎉 Predicted Revenue: **${prediction:.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Additional info
st.markdown("""
---
**Note:** This model uses advanced features like interaction terms and a Random Forest algorithm for improved accuracy.  
""")
