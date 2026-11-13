
from pathlib import Path
from datetime import date

import joblib
import pandas as pd
import streamlit as st

# =========================
# 1. Paths
# =========================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "late_delivery_pipeline_v2.joblib"

# =========================
# 2. Page Configuration
# =========================

st.set_page_config(
    page_title="Olist Delivery Risk",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Late-Delivery Risk Predictor")

st.caption(
    "Machine Learning system for predicting e-commerce delivery delays."
)

st.divider()

# =========================
# 3. Load Model
# =========================

if not MODEL_PATH.exists():
    st.error("Model file not found. Run 05_model.ipynb first.")
    st.stop()


@st.cache_resource
def load_model():

    artifact = joblib.load(MODEL_PATH)

    return (
        artifact["pipeline"],
        artifact["threshold"],
        artifact["feature_columns"]
    )


model, threshold, feature_columns = load_model()

# =========================
# 4. Order Information
# =========================

st.header("Order Information")

st.write(
    "Enter order details to estimate the probability of late delivery."
)

with st.form("order_details"):

    col1, col2 = st.columns(2)

    # =========================
    # Order Information
    # =========================

    with col1:

        st.subheader("Product & Payment")

        category = st.selectbox(
            "Product Category",
            [
                "bed_bath_table",
                "health_beauty",
                "sports_leisure",
                "furniture_decor",
                "computers_accessories",
                "housewares",
                "watches_gifts",
                "toys",
                "cool_stuff",
                "other"
            ]
        )

        price = st.number_input(
            "Products Total Price (R$)",
            min_value=0.0,
            value=120.0
        )

        freight = st.number_input(
            "Freight Total (R$)",
            min_value=0.0,
            value=20.0
        )

        item_count = st.number_input(
            "Number of Items",
            min_value=1,
            max_value=30,
            value=1
        )

        installments = st.number_input(
            "Payment Installments",
            min_value=1,
            max_value=24,
            value=1
        )

        payment_type = st.selectbox(
            "Payment Type",
            [
                "credit_card",
                "boleto",
                "voucher",
                "debit_card"
            ]
        )

    # =========================
    # Delivery Information
    # =========================

    with col2:

        st.subheader("Delivery Information")

        seller_id = st.text_input(
            "Seller ID",
            value="unknown"
        )

        states = [
            "SP", "RJ", "MG", "RS", "PR", "SC", "BA",
            "DF", "GO", "ES", "PE", "CE", "PA", "MT",
            "MS", "PB", "RN", "AL", "PI", "SE", "TO",
            "RO", "AM", "AC", "AP", "RR", "MA"
        ]

        customer_state = st.selectbox(
            "Customer State",
            states
        )

        seller_state = st.selectbox(
            "Seller State",
            states
        )

        distance = st.number_input(
            "Approximate Distance (km)",
            min_value=0.0,
            value=350.0
        )

        delivery_days = st.number_input(
            "Estimated Delivery Window (days)",
            min_value=1.0,
            max_value=90.0,
            value=15.0
        )

        purchase_date = st.date_input(
            "Purchase Date",
            value=date.today()
        )

        hour = st.slider(
            "Purchase Hour",
            0,
            23,
            14
        )

    submitted = st.form_submit_button(
        "Predict Late Delivery Risk",
        use_container_width=True
    )

# =========================
# 5. Prediction
# =========================

if submitted:

    month = purchase_date.month
    weekday = purchase_date.weekday()
    year = purchase_date.year
    day = purchase_date.day

    quarter = (month - 1) // 3 + 1

    week_of_year = purchase_date.isocalendar().week

    day_names = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    month_names = [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ]

    day_period = (
        "Night" if hour < 6 else
        "Morning" if hour < 12 else
        "Afternoon" if hour < 18 else
        "Evening"
    )

    # =========================
    # Input Features
    # =========================

    input_values = {

        "total_price": price,
        "total_freight": freight,
        "item_count": item_count,
        "payment_value": price + freight,
        "payment_installments": installments,

        "distance_km": distance,
        "estimated_delivery_days": delivery_days,

        "freight_share": freight / price if price > 0 else 0,
        "average_item_price": price / item_count,
        "average_freight_per_item": freight / item_count,

        "purchase_year": year,
        "purchase_quarter": quarter,
        "purchase_month": month,
        "purchase_day": day,
        "purchase_week_of_year": week_of_year,
        "purchase_day_of_week": weekday,
        "purchase_hour": hour,

        "Year": year,
        "Quarter": quarter,
        "Month": month,
        "Day": day,
        "Hour": hour,
        "WeekOfYear": week_of_year,
        "DayOfWeek": weekday,
        "DayName": day_names[weekday],
        "MonthName": month_names[month - 1],
        "Day_Period": day_period,

        "is_weekend": int(weekday >= 5),
        "IsWeekend": int(weekday >= 5),
        "is_evening": int(hour >= 18),

        "payment_type": payment_type,
        "product_category_name": category,
        "seller_id": seller_id.strip() or "unknown",
        "customer_state": customer_state,
        "seller_state": seller_state
    }

    # Match training feature columns
    row = pd.DataFrame([input_values]).reindex(
        columns=feature_columns
    )

    missing_features = [
        col for col in feature_columns
        if col not in input_values
    ]

    if missing_features:

        st.warning(
            "Some training features are not provided by the form. "
            "Their values may be imputed by the preprocessing pipeline."
        )

        with st.expander("Missing Features"):
            st.write(missing_features)

    # =========================
    # Model Prediction
    # =========================

    try:

        probability = float(
            model.predict_proba(row)[0, 1]
        )

        prediction = int(
            probability >= threshold
        )

        st.divider()

        st.header("Prediction Results")

        col1, col2 = st.columns(2)

        col1.metric(
            "Estimated Late Risk",
            f"{probability:.1%}"
        )

        col2.metric(
            "Decision Threshold",
            f"{threshold:.2f}"
        )

        st.progress(
            min(max(probability, 0.0), 1.0)
        )

        if prediction == 1:

            st.error(
                "⚠️ The model predicts that this order may be late."
            )

        else:

            st.success(
                "✅ The model predicts that this order may arrive on time."
            )

        st.caption(
            "This prediction is based on historical data and is not a guarantee."
        )

    except Exception as e:

        st.error("Prediction failed.")
        st.exception(e)

# =========================
# 6. Footer
# =========================

st.divider()

st.caption(
    "Olist Late-Delivery Risk Predictor | Machine Learning Project"
)
