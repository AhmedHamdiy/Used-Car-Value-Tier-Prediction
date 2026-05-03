import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib
from datetime import datetime

@st.cache_resource
def load_model():
    # Update this path if your model is inside a specific folder, e.g., "models/model.pkl"
    model = joblib.load("models/best_model_for_production.pkl")
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

@st.cache_data
def load_model_comparisons():
    # Update the path if your CSV is in a different folder
    df = pd.read_csv("reports/model_comparison.csv")
    return df

# Load the comparison data alongside your other data
df = load_data()
model = load_model()
fi_df = load_feature_importance()
comp_df = load_model_comparisons()

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
    current_year = datetime.now().year
    df_processed['vehicleAge'] = current_year - user_input['yearOfRegistration']
    df_processed['kmPerYear'] = user_input['kilometer'] / (df_processed['vehicleAge'] + 1)
    df_processed['km_power_ratio'] = user_input['kilometer'] / (user_input['power'] + 1)
    df_processed['log_km'] = np.log1p(user_input['kilometer'])

    # Frequency Encodings - computed from training data
    BRAND_FREQ = {
        "volkswagen": 0.200091,
        "bmw": 0.113960,
        "mercedes-benz": 0.104804,
        "opel": 0.096641,
        "audi": 0.096032,
        "ford": 0.066154,
        "renault": 0.043726,
        "peugeot": 0.030603,
        "fiat": 0.024497,
        "seat": 0.019900,
        "skoda": 0.017669,
        "mazda": 0.016253,
        "smart": 0.015558,
        "citroen": 0.015180,
        "toyota": 0.014748,
        "nissan": 0.013821,
        "hyundai": 0.011259,
        "mini": 0.010954,
        "volvo": 0.009939,
        "mitsubishi": 0.008680,
    }

    MODEL_FREQ = {
        "golf": 0.082492,
        "3-series": 0.060646,
        "c-class": 0.037851,
        "a4": 0.036454,
        "corsa": 0.031765,
        "polo": 0.028268,
        "astra": 0.028048,
        "passat": 0.026612,
        "5-series": 0.024408,
        "focus": 0.022637,
        "e-class": 0.022289,
        "a3": 0.019615,
        "2-reihe": 0.018607,
        "a6": 0.018021,
        "transporter": 0.016307,
        "fiesta": 0.014207,
        "twingo": 0.013845,
        "fortwo": 0.013779,
        "punto": 0.013019,
        "a-class": 0.012424,
    }

    df_processed['brand_freq'] = user_input['brand'].map(BRAND_FREQ).fillna(0.005)
    df_processed['model_freq'] = 0.01  # Default - model not in sidebar

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
    # Take a maximum of 1,500 points. This keeps the dashboard lightning fast
    # and prevents the browser's graphing engine from silently crashing.
    sample_size = min(1500, len(plot_df))
    plot_df = plot_df.sample(n=sample_size, random_state=42)

    try:
        # 3. Create the chart
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

        # 4. Render the chart
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

# -----------------------------------------------------------------------------
# 7. Model Selection & Business Performance
# -----------------------------------------------------------------------------
st.divider()
st.subheader("🏆 Model Selection & Business Impact")

st.markdown("""
To ensure the most accurate pricing predictions, we evaluated several machine learning algorithms.
Beyond standard accuracy, we optimized for two critical business metrics:
* **Luxury Detection Rate:** The model's ability to successfully identify high-margin premium vehicles.
* **Critical Error Rate:** The percentage of severe pricing mistakes (e.g., misclassifying a 'Luxury' car as 'Budget').
""")

# 1. Map technical terms to business-friendly language
model_map = {
    'xgboost': 'XGBoost (Advanced Ensemble)',
    'decision_tree': 'Decision Tree',
    'random_forest': 'Random Forest',
    'log_reg': 'Logistic Regression',
    'baseline': 'Baseline (Manual Rule)'
}
data_map = {
    'none': 'Standard',
    'smote': 'AI Augmented (SMOTE)',
    'undersample': 'Balanced Data'
}

# 2. Apply mappings and convert decimals to percentages
comp_df['Algorithm'] = comp_df['model'].map(model_map)
comp_df['Data Strategy'] = comp_df['dataset'].map(data_map)
comp_df['Overall Accuracy'] = comp_df['test_f1'] * 100
comp_df['Luxury Detection Rate'] = comp_df['luxury_recall'] * 100
comp_df['Critical Error Rate'] = comp_df['severe_misclassification_rate'] * 100

# 3. Sort so the best models appear first
comp_df = comp_df.sort_values('Overall Accuracy', ascending=False)

# Create two columns layout
col_chart, col_table = st.columns([1.2, 1])

with col_chart:
    # Grouped bar chart comparing the algorithms
    fig_comp = px.bar(
        comp_df,
        x="Algorithm",
        y="Overall Accuracy",
        color="Data Strategy",
        barmode="group",
        title="Predictive Accuracy by Algorithm",
        labels={"Overall Accuracy": "Overall Accuracy (%)"},
        text_auto='.1f' # Display 1 decimal place on bars
    )

    # Set Y-axis to 100% and move the legend inside the chart to save space
    fig_comp.update_layout(
        yaxis_range=[0, 100],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_comp, width='stretch', key="model_comparison_chart")

with col_table:
    st.write("**Detailed Business Metrics**")

    # Isolate only the clean, business-friendly columns for the table
    display_df = comp_df[['Algorithm', 'Data Strategy', 'Overall Accuracy', 'Luxury Detection Rate', 'Critical Error Rate']].copy()

    # Format as clean string percentages (e.g., "85.9%")
    display_df['Overall Accuracy'] = display_df['Overall Accuracy'].map("{:.1f}%".format)
    display_df['Luxury Detection Rate'] = display_df['Luxury Detection Rate'].map("{:.1f}%".format)
    display_df['Critical Error Rate'] = display_df['Critical Error Rate'].map("{:.2f}%".format)

    # Display the final pristine table
    st.dataframe(display_df, width='stretch', hide_index=True)
