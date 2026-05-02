import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Used Car Value Tier Predictor", layout="wide")

# Use st.cache_data so we don't reload the CSV and Model on every single button click
@st.cache_data
def load_data():
    # Replace with your actual cleaned data path
    return pd.read_csv("data/processed/clean_data.csv")

@st.cache_resource
def load_model():
    # Replace with the actual path to your trained model
    # return joblib.load("models/best_car_tier_model.joblib")
    pass 

df = load_data()
# model = load_model()

# -----------------------------------------------------------------------------
# 2. The Sidebar: Interactive "What-If" Controls
# -----------------------------------------------------------------------------
st.sidebar.header("Vehicle Specifications")
st.sidebar.write("Adjust the parameters below to see how they impact the predicted value tier.")

# Create input widgets based on your cleaned data schema
input_brand = st.sidebar.selectbox("Brand", df['brand'].unique())
input_power = st.sidebar.slider("Horsepower (PS)", min_value=50, max_value=500, value=120)
input_km = st.sidebar.slider("Kilometers Driven", min_value=0, max_value=300000, step=10000, value=50000)
input_year = st.sidebar.slider("Year of Registration", min_value=1990, max_value=2024, value=2015)
input_gearbox = st.sidebar.selectbox("Gearbox", ["manual", "automatic"])

# -----------------------------------------------------------------------------
# 3. Main Dashboard: Executive Summary
# -----------------------------------------------------------------------------
st.title("🚗 Used Car Value Tier Dashboard")
st.markdown("""
This tool allows stakeholders to instantly classify incoming vehicle inventory into **Budget**, **Mid-Range**, or **Luxury** tiers based on historical market data and our machine learning model.
""")

st.divider()

# Create 3 columns for high-level business metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total Vehicles in Database", value=f"{len(df):,}")
with col2:
    st.metric(label="Average Market Price", value=f"€{df['price'].mean():,.0f}")
with col3:
    # Example insight: Percentage of cars that are luxury
    lux_pct = (len(df[df['price_tier'] == 'luxury']) / len(df)) * 100
    st.metric(label="Luxury Market Share", value=f"{lux_pct:.1f}%")

# -----------------------------------------------------------------------------
# 4. Interactive Prediction Engine
# -----------------------------------------------------------------------------
st.subheader("🔮 Live Tier Prediction")

# Format the user's input into a DataFrame that matches what your model expects
user_input = pd.DataFrame({
    'brand': [input_brand],
    'power': [input_power],
    'kilometer': [input_km],
    'yearOfRegistration': [input_year],
    'gearbox': [input_gearbox]
})

# Make the prediction (commented out until you link your real model)
# prediction = model.predict(user_input)[0]
prediction = "Mid-Range"  # Placeholder

st.success(f"Based on the selected specifications, this vehicle is classified as: **{prediction.upper()}**")

# -----------------------------------------------------------------------------
# 5. EDA Findings (Visual Context for Stakeholders)
# -----------------------------------------------------------------------------
st.divider()
st.subheader("📊 Market Context")
st.write(f"How does the selected **{input_brand.title()}** compare to the rest of the market?")

# Create an interactive Plotly chart
fig = px.scatter(
    df[df['brand'] == input_brand], 
    x="kilometer", 
    y="price", 
    color="price_tier",
    title=f"Price Depreciation by Mileage for {input_brand.title()}s",
    labels={"kilometer": "Kilometers Driven", "price": "Price (€)", "price_tier": "Value Tier"},
    opacity=0.6
)


st.plotly_chart(fig, use_container_width=True)