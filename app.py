import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import pickle
# OR import joblib (depending on how you saved it)

@st.cache_resource
def load_model():
    # Update this path if your model is inside a specific folder, e.g., "models/model.pkl"
    with open("models/best_model_for_production.pkl", "rb") as file:
        model = pickle.load(file)
    return model


st.set_page_config(page_title="Used Car Value Tier Predictor", layout="wide")

# Use st.cache_data so we don't reload the CSV and Model on every single button click
@st.cache_data
def load_data():
    # Replace with your actual cleaned data path
    return pd.read_csv("data/processed/clean_data.csv")

@st.cache_data
def load_feature_importance():
    # Update the path if your CSV is in a different folder
    df = pd.read_csv("reports/results/feature_importance.csv")

    # Sort the values so the most important features appear at the top of the chart
    # We sort ascending=True because Plotly builds horizontal bar charts from the bottom up
    return df.sort_values(by="importance", ascending=True)


df = load_data()
model = load_model()
fi_df = load_feature_importance()

# -----------------------------------------------------------------------------
# 2. The Sidebar: Interactive "What-If" Controls
# -----------------------------------------------------------------------------
st.sidebar.header("Vehicle Specifications")
st.sidebar.write("Adjust the parameters below to see how they impact the predicted value tier.")

# CHANGE THE TEXT IN THE QUOTES TO RENAME THE FIELDS
input_brand = st.sidebar.selectbox("Vehicle Make", df['brand'].unique())
input_power = st.sidebar.slider("Engine Power (Horsepower)", min_value=10, max_value=500, value=120)
input_km = st.sidebar.slider("Current Mileage (km)", min_value=0, max_value=300000, step=10000, value=50000)
input_year = st.sidebar.slider("Registration Year", min_value=1970, max_value=2026, value=2015)
input_gearbox = st.sidebar.selectbox("Transmission Type", ["manual", "automatic"])

# -----------------------------------------------------------------------------
# 3. Main Dashboard: Executive Summary
# -----------------------------------------------------------------------------
st.title("🚗 Used Car Value Tier Dashboard")
st.markdown("""
This tool allows stakeholders to instantly classify incoming vehicle inventory into **Budget**,
            **Mid-Range**, or **Luxury** tiers based on historical market data and our machine learning model.
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

# 1. Capture raw user input
# Notice we need to gather all the inputs required to build the 24 features
user_input = pd.DataFrame({
    'brand': [input_brand],
    'power': [input_power],
    'kilometer': [input_km],
    'yearOfRegistration': [input_year],
    'gearbox': [input_gearbox],
    # Add dummy inputs for fields your model expects but you might not have in the sidebar yet
    'vehicleType': ['sedan'], # Default or add a selectbox for this
    'fuelType': ['gasoline'], # Default or add a selectbox for this
    'seller': ['dealer'],     # Default or add a selectbox for this
    'dataSource': ['kaggle']  # Default
})

labels_map ={
    0: 'budget',
    1: 'mid-range',
    2: 'luxury'
}
try:
    # 2. FEATURE ENGINEERING (Replicating your notebook logic)
    df_processed = pd.DataFrame()

    # Numeric passthroughs
    df_processed['power'] = user_input['power']
    df_processed['kilometer'] = user_input['kilometer']

    # Engineered Features
    current_year = 2026 # Or whatever year you used in training
    df_processed['vehicleAge'] = current_year - user_input['yearOfRegistration']
    df_processed['kmPerYear'] = user_input['kilometer'] / (df_processed['vehicleAge'] + 1)
    df_processed['km_power_ratio'] = user_input['kilometer'] / (user_input['power'] + 1)
    df_processed['log_km'] = np.log1p(user_input['kilometer'])

    # Frequency Encodings (You MUST use the exact same frequencies from your training data)
    # If you didn't save these mappings, you'll need to use placeholder averages for now,
    # but ideally, you save a dictionary of these frequencies during training and load them here.
    brand_freq_map = {'volkswagen': 0.2, 'bmw': 0.1} # REPLACE WITH REAL MAPPINGS
    model_freq_map = {'golf': 0.05, '3-series': 0.03} # REPLACE WITH REAL MAPPINGS

    df_processed['brand_freq'] = user_input['brand'].map(brand_freq_map).fillna(0.01)
    # Note: If you don't have model input, you'll need to add it or use a default
    df_processed['model_freq'] = 0.01 # Placeholder if model input isn't collected

    # 3. ONE-HOT ENCODING (Manual implementation to guarantee column names match)
    # We must explicitly create ALL expected dummy columns and set them to 0 or False initially

    # List of expected OHE columns from the error message
    ohe_columns = [
        'vehicleType_convertible', 'vehicleType_coupe', 'vehicleType_other',
        'vehicleType_sedan', 'vehicleType_station-wagon', 'vehicleType_suv', 'vehicleType_van',
        'gearbox_manual', 'gearbox_semi-automatic',
        'fuelType_diesel', 'fuelType_electric', 'fuelType_gasoline', 'fuelType_hybrid', 'fuelType_lpg',
        'seller_private', 'dataSource_kaggle'
    ]

    # Initialize all OHE columns to False (or 0)
    for col in ohe_columns:
        df_processed[col] = False

    # Set the appropriate column to True based on user input
    # Gearbox
    if f"gearbox_{input_gearbox}" in df_processed.columns:
        df_processed[f"gearbox_{input_gearbox}"] = True

    # Vehicle Type (Assuming you add input_vehicleType to sidebar)
    # if f"vehicleType_{input_vehicleType}" in df_processed.columns:
    #     df_processed[f"vehicleType_{input_vehicleType}"] = True

    # Fuel Type (Assuming you add input_fuelType to sidebar)
    # if f"fuelType_{input_fuelType}" in df_processed.columns:
    #     df_processed[f"fuelType_{input_fuelType}"] = True

    # Seller
    # if user_input['seller'].iloc[0] == 'private':
    #     df_processed['seller_private'] = True

    # DataSource
    df_processed['dataSource_kaggle'] = True

    # 4. Enforce exact column order required by the model
    expected_cols = [
        'power', 'kilometer', 'vehicleAge', 'kmPerYear', 'km_power_ratio',
        'log_km', 'brand_freq', 'model_freq', 'vehicleType_convertible',
        'vehicleType_coupe', 'vehicleType_other', 'vehicleType_sedan',
        'vehicleType_station-wagon', 'vehicleType_suv', 'vehicleType_van',
        'gearbox_manual', 'gearbox_semi-automatic', 'fuelType_diesel',
        'fuelType_electric', 'fuelType_gasoline', 'fuelType_hybrid',
        'fuelType_lpg', 'seller_private', 'dataSource_kaggle'
    ]

    # Reorder the dataframe
    df_processed = df_processed[expected_cols]

    # 5. Make Prediction
    prediction_array = model.predict(df_processed)
    prediction = str(prediction_array[0])

    st.success(f"Based on the selected specifications, this vehicle is classified as: **{labels_map.get(int(prediction), 'Unknown')}**")

except Exception as e:
    st.error(f"Prediction Error: {e}")

# -----------------------------------------------------------------------------
# 5. EDA Findings (Visual Context for Stakeholders)
# -----------------------------------------------------------------------------
st.divider()
st.subheader("📊 Market Context")
st.write(f"How does the selected **{input_brand.title()}** compare to the rest of the market?")

# 1. Filter the data for the selected brand
plot_df = df[df['brand'] == input_brand]

# 2. Safety Check: Is the filtered data empty?
if plot_df.empty:
    st.warning(f"No pricing data available in the dataset for {input_brand.title()}.")
else:
    # 3. THE FIX: Sample the data!
    # Take a maximum of 1,500 points. This keeps the dashboard lightning fast
    # and prevents the browser's graphing engine from silently crashing.
    sample_size = min(1500, len(plot_df))
    plot_df = plot_df.sample(n=sample_size, random_state=42)

    try:
        # 4. Create the chart
        # Note: If you renamed these columns in Phase 1 (e.g., to 'kilometers' or 'price_log'),
        # update x="kilometer" and y="price" to match your CSV perfectly.
        fig = px.scatter(
            plot_df,
            x="kilometer",
            y="price",
            color="price_tier",
            title=f"Price Depreciation by Mileage for {input_brand.title()}s (Sampled)",
            labels={"kilometer": "Kilometers Driven", "price": "Price (€)", "price_tier": "Value Tier"},
            opacity=0.6
        )

        # 5. Render the chart
        st.plotly_chart(fig, width='stretch', key="unique_chart_name_here")

    except Exception as e:
        st.error(f"Could not render the chart. Make sure the 'kilometer', 'price', and 'price_tier' columns exist in your CSV. Error: {e}")

# -----------------------------------------------------------------------------
# 6. Model Insights: What drives the price?
# -----------------------------------------------------------------------------
st.divider()
st.subheader("🧠 Behind the Curtain: What drives car value?")

# Add a stakeholder-friendly explanation
st.markdown("""
Our machine learning model analyzes multiple factors to determine a vehicle's value tier.
The chart below ranks these factors by their overall impact.
The longer the bar, the more heavily that specific attribute influences the final price prediction.
""")
# Clean up technical feature names for the stakeholders
name_mapping = {
    "yearOfRegistration": "Registration Year",
    "kilometer": "Kilometers Driven",
    "power": "Horsepower (PS)",
    "vehicleType_suv": "Is SUV",
    "gearbox_automatic": "Has Automatic Gearbox"
}
fi_df["feature"] = fi_df["feature"].replace(name_mapping)

# Create a horizontal bar chart using Plotly
fig_importance = px.bar(
    fi_df,
    x="importance",
    y="feature",
    orientation="h",
    color="importance",
    color_continuous_scale="Blues", # Gives it a professional, sleek look
    title="Top Factors Influencing Vehicle Price Tiers",
    labels={
        "importance": "Relative Impact on Price",
        "feature": "Vehicle Attribute"
    }
)

# Hide the color scale bar on the side, as it's redundant and clutters the UI
fig_importance.update_layout(coloraxis_showscale=False)

# Render the chart
st.plotly_chart(fig_importance, width='stretch')