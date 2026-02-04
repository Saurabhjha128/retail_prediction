import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Retail Spend Predictor",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 Retail Spend Prediction")
st.write("Enter customer transaction details to predict future spend")

st.divider()

# Input fields
spend_30 = st.number_input("Spend in past 30 days", min_value=0.0)
orders_30 = st.number_input("Orders in past 30 days", min_value=0)

spend_60 = st.number_input("Spend in past 60 days", min_value=0.0)
orders_60 = st.number_input("Orders in past 60 days", min_value=0)

spend_90 = st.number_input("Spend in past 90 days", min_value=0.0)
orders_90 = st.number_input("Orders in past 90 days", min_value=0)

spend_180 = st.number_input("Spend in past 180 days", min_value=0.0)
orders_180 = st.number_input("Orders in past 180 days", min_value=0)

recency = st.number_input("Recency (days since last purchase)", min_value=0)

# Button
if st.button("🔮 Predict Spend"):
    payload = {
        "spend_past_30d": spend_30,
        "orders_past_30d": orders_30,
        "spend_past_60d": spend_60,
        "orders_past_60d": orders_60,
        "spend_past_90d": spend_90,
        "orders_past_90d": orders_90,
        "spend_past_180d": spend_180,
        "orders_past_180d": orders_180,
        "recency_days": recency
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        preds = response.json()
        st.success("✅ Prediction Successful!")

        col1, col2 = st.columns(2)
        col1.metric("Next 30 Days", f"₹ {preds['30d']:.2f}")
        col2.metric("Next 60 Days", f"₹ {preds['60d']:.2f}")

        col3, col4 = st.columns(2)
        col3.metric("Next 90 Days", f"₹ {preds['90d']:.2f}")
        col4.metric("Next 180 Days", f"₹ {preds['180d']:.2f}")

    else:
        st.error("❌ API Error. Is FastAPI running?")
