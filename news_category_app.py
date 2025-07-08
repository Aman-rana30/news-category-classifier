import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
from PIL import Image


@st.cache_resource
def load_model():
    model_path = "logreg_model.pkl"
    vectorizer_path = "tfidf_vectorizer.pkl"
    label_encoder_path = "label_encoder.pkl"
   
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    
    return model, vectorizer, label_encoder

# Initialize Streamlit app
st.set_page_config(page_title="News Category Classifier", page_icon="📰", layout="wide")

# Sidebar
st.sidebar.header("Options")
show_examples = st.sidebar.checkbox("Show Sample Headlines", value=True)
show_category_info = st.sidebar.checkbox("Show Category Information", value=True)

# Main content
st.title("News Category Classifier 📰")

# App description
st.markdown("""
    <style>
    .description {
        font-size: 18px;
        line-height: 1.6;
        padding: 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
    
    <div class='description' style='color: #333;'>
    🌟 Enter a news headline below and our AI model will predict its category with confidence scores. 
    Try different headlines to see how well the model performs!
    </div>
    
    <style>
    .stTextInput input {
        font-size: 16px;
        padding: 12px;
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    .stTextInput input:focus {
        border-color: #4CAF50;
        box-shadow: 0 0 5px rgba(76, 175, 80, 0.3);
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #45a049;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# Input section with examples
with st.form(key='news_form'):
    col1, col2 = st.columns([3, 1])
    
    with col1:
        headline = st.text_input("Enter News Headline", "", placeholder="Enter a news headline here...")
        submit_button = st.form_submit_button(label='Classify', help="Click to predict the news category")
    
    with col2:
        if show_examples:
            st.write("Sample Headlines:")
            st.write("- Breaking: New Study Shows...")
            st.write("- Government Announces...")
            st.write("- Market Analysis:...")

# Process and display results
if submit_button:
    if headline:
        try:
            # Load model and vectorizer
            model, vectorizer, label_encoder = load_model()
            
            # Preprocess input
            headline_vector = vectorizer.transform([headline])
            
            # Make prediction
            prediction = model.predict(headline_vector)[0]
            predicted_category = label_encoder.inverse_transform([prediction])[0]
            
            # Display prediction with confidence
            st.markdown("""
                <style>
                .prediction-box {
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;
                    border-left: 4px solid #4CAF50;
                    margin: 20px 0;
                }
                .prediction-text {
                    font-size: 24px;
                    font-weight: bold;
                    color: #333;
                }
                </style>
                """, unsafe_allow_html=True)
            
            st.markdown("""
                <div class="prediction-box">
                    <div class="prediction-text">
                        Predicted Category: {predicted_category}
                    </div>
                </div>
                """.format(predicted_category=predicted_category), unsafe_allow_html=True)
            
            # Show confidence scores with visualization
            probabilities = model.predict_proba(headline_vector)[0]
            confidence_scores = dict(zip(label_encoder.classes_, probabilities))
            confidence_scores = {k: v for k, v in sorted(confidence_scores.items(), key=lambda item: item[1], reverse=True)}
            
            # Create a bar chart for confidence scores
            categories = list(confidence_scores.keys())
            scores = list(confidence_scores.values())
            
            fig = px.bar(x=categories, y=scores, 
                        labels={'x': 'Categories', 'y': 'Confidence Score'},
                        title='Confidence Scores for Each Category')
            fig.update_layout(
                xaxis_tickangle=-45,
                showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show raw scores
            st.write("Raw Confidence Scores:")
            for category, score in confidence_scores.items():
                st.write(f"- {category}: {score:.2%}")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please try again with a different headline.")
    else:
        st.warning("Please enter a news headline")

# Add category information
if show_category_info:
    st.markdown("""
        <style>
        .category-info {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        .category-title {
            font-size: 20px;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="category-info">
            <div class="category-title">Category Information</div>
            <p>Our model can classify news into several categories:</p>
            <ul>
                <li>Politics: Government news, policy updates, political events</li>
                <li>Business: Market analysis, company news, economic reports</li>
                <li>Technology: Innovation, gadgets, software updates</li>
                <li>Sports: Sports events, athlete news, tournament updates</li>
                <li>Entertainment: Movies, music, celebrity news</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Add footer with social links
st.markdown("""
    <style>
    .footer {
        margin-top: 40px;
        padding: 20px;
        text-align: center;
        background-color: #f0f2f6;
        border-radius: 10px;
    }
    .footer-links {
        font-size: 16px;
        color: #666;
    }
    .footer-links a {
        color: #4CAF50;
        text-decoration: none;
        margin: 0 10px;
    }
    .footer-links a:hover {
        text-decoration: underline;
    }
    </style>
    
    <div class="footer">
        <div class="footer-links">
            Created with ❤️ using Streamlit • 
            <a href="https://github.com/Aman-rana30" target="_blank">GitHub</a> • 
            <a href="https://www.linkedin.com/in/amandeep-singh-268bb8308" target="_blank">LinkedIn</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
