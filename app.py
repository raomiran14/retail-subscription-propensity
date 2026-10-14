import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Page Config (Called exactly once at the top)
st.set_page_config(
    page_title="Customer Subscription Propensity Engine",
    page_icon="🛍️",
    layout="wide"
)

# 1. Load Trained Pipeline Artifact from models/
@st.cache_resource
def load_model():
    return joblib.load('models/subscription_model_pipeline.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model pipeline: {e}")
    st.info("Ensure 'subscription_model_pipeline.pkl' is placed inside the 'models/' folder.")
    st.stop()

# 2. Application Header
st.title("🛍️ Retail Customer Subscription Propensity Scorecard")
st.markdown("""
This application uses Machine Learning to predict whether a shopper is likely to convert to a **Premium Subscription**, 
helping marketing teams deliver targeted incentives rather than blanket discounts.
""")

st.divider()

# 3. Sidebar Input Form
st.sidebar.header("📋 Customer Profile Data")

# Input Controls matching raw dataset columns
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
category = st.sidebar.selectbox("Product Category", ["Clothing", "Footwear", "Outerwear", "Accessories"])
item_purchased = st.sidebar.text_input("Item Purchased", value="Blouse")
purchase_amount = st.sidebar.number_input("Purchase Amount ($)", min_value=5.0, max_value=500.0, value=65.0)
location = st.sidebar.selectbox("State / Location", ["Kentucky", "Maine", "Massachusetts", "Rhode Island", "Oregon", "California", "New York"])
size = st.sidebar.selectbox("Size", ["S", "M", "L", "XL"])
color = st.sidebar.selectbox("Color", ["Gray", "Maroon", "Turquoise", "Black", "White"])
season = st.sidebar.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])
review_rating = st.sidebar.slider("Review Rating", min_value=1.0, max_value=5.0, value=3.8, step=0.1)
shipping_type = st.sidebar.selectbox("Shipping Type", ["Express", "Free Shipping", "Next Day Air", "Standard", "Store Pickup", "2-Day Shipping"])
discount_applied = st.sidebar.radio("Discount Applied?", ["Yes", "No"])
previous_purchases = st.sidebar.number_input("Previous Purchases Count", min_value=0, max_value=100, value=12)
payment_method = st.sidebar.selectbox("Payment Method", ["Venmo", "Cash", "Credit Card", "PayPal", "Debit Card"])
frequency_of_purchases = st.sidebar.selectbox("Frequency of Purchases", ["Weekly", "Fortnightly", "Monthly", "Annually", "Bi-Weekly", "Quarterly", "Every 3 Months"])

# Helper function to map frequency to days
freq_days_map = {
    "Weekly": 7, "Fortnightly": 14, "Bi-Weekly": 14,
    "Monthly": 30, "Quarterly": 90, "Every 3 Months": 90, "Annually": 365
}
purchase_frequency_days = freq_days_map.get(frequency_of_purchases, 30)

# Map age to age_group matching Notebook labels
if age < 30:
    age_group = "Young Adult"
elif age < 45:
    age_group = "Adult"
elif age < 60:
    age_group = "Middle-aged"
else:
    age_group = "Senior"

# 4. Construct DataFrame for Model
input_df = pd.DataFrame([{
    'age': age,
    'gender': gender,
    'item_purchased': item_purchased,
    'category': category,
    'purchase_amount': purchase_amount,
    'location': location,
    'size': size,
    'color': color,
    'season': season,
    'review_rating': review_rating,
    'shipping_type': shipping_type,
    'discount_applied': discount_applied,
    'previous_purchases': previous_purchases,
    'payment_method': payment_method,
    'frequency_of_purchases': frequency_of_purchases,
    'age_group': age_group,
    'purchase_frequency_days': purchase_frequency_days
}])

# 5. Model Inference & Prediction Output
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔍 Prediction Results")
    
    # Predict Class and Probability Score
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] # Probability of Class 1 (Yes)

    # Scorecard Display
    st.metric(
        label="Subscription Conversion Probability",
        value=f"{probability * 100:.1f}%",
        delta="High Propensity" if probability >= 0.50 else "Low Propensity"
    )

    if prediction == 1:
        st.success("🎯 **Result:** High Probability Subscriber (Likely to Join)")
    else:
        st.warning("⚠️ **Result:** Standard Customer (Low Conversion Propensity)")

with col2:
    st.subheader("📊 Key Customer Indicators")
    st.write(f"**Discount Used:** {discount_applied}")
    st.write(f"**Customer Loyalty Tier:** {previous_purchases} prior orders")
    st.write(f"**Average Order Spend:** ${purchase_amount:.2f}")
    st.write(f"**Shopping Cadence:** {frequency_of_purchases} ({purchase_frequency_days} days)")

st.divider()