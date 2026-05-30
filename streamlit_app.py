import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("random_forest_model.pkl", "rb") as file:
    model_data = pickle.load(file)

model = model_data["model"]

# Page settings
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Title
st.title("🏠 House Price Predictor")
st.write("Enter property details to estimate the house price in Indian Rupees (₹).")


def format_inr(value):
    if value >= 10000000:
        return f"₹{value / 10000000:.2f} Crore"
    elif value >= 100000:
        return f"₹{value / 100000:.2f} Lakh"
    else:
        return f"₹{value:,.2f}"


# Input widgets with unique keys
area = st.number_input(
    "Total Area (sq ft)",
    min_value=100,
    value=1200,
    key="area"
)

bedrooms = st.number_input(
    "Bedrooms",
    min_value=1,
    value=3,
    step=1,
    key="bedrooms"
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=1,
    value=2,
    step=1,
    key="bathrooms"
)

floors = st.number_input(
    "Floors",
    min_value=1,
    value=1,
    step=1,
    key="floors"
)

year_built = st.number_input(
    "Year Built",
    min_value=1900,
    max_value=2100,
    value=2015,
    step=1,
    key="year_built"
)

location = st.selectbox(
    "Location",
    ["Downtown", "Suburban", "Rural"],
    key="location"
)

condition = st.selectbox(
    "Condition",
    ["Excellent", "Good", "Fair"],
    key="condition"
)

garage = st.selectbox(
    "Garage",
    ["Yes", "No"],
    key="garage"
)

# Predict button
if st.button("Predict Price", key="predict_button"):
    input_df = pd.DataFrame([{
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Floors": floors,
        "YearBuilt": year_built,
        "Location": location,
        "Condition": condition,
        "Garage": garage
    }])

    prediction = model.predict(input_df)[0]

    st.success(f"Estimated House Price: {format_inr(prediction)}")
