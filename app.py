"""Small Streamlit interface for trying the trained churn model."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("artifacts/churn_model.joblib")

st.set_page_config(page_title="Churn prediction", page_icon="📉", layout="centered")
st.title("Customer churn prediction")
st.write("Enter a customer profile to estimate their likelihood of leaving.")

if not MODEL_PATH.exists():
    st.error("The model has not been trained. Run the training command from the README first.")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.form("customer"):
    left, right = st.columns(2)
    with left:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.number_input("Monthly charges", 0.0, 200.0, 70.0)
        usage = st.number_input("Monthly usage (GB)", 0.0, 500.0, 120.0)
        support = st.slider("Support calls in 6 months", 0, 10, 1)
        late = st.slider("Late payments in 12 months", 0, 12, 0)
        contract = st.selectbox("Contract", ["month-to-month", "one-year", "two-year"])
    with right:
        internet = st.selectbox("Internet service", ["fiber", "dsl", "none"])
        payment = st.selectbox("Payment method", ["electronic-check", "bank-transfer", "credit-card", "mailed-check"])
        senior = st.selectbox("Senior citizen", ["no", "yes"])
        partner = st.selectbox("Has partner", ["no", "yes"])
        paperless = st.selectbox("Paperless billing", ["yes", "no"])
    submitted = st.form_submit_button("Estimate churn risk", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "total_charges": monthly * tenure,
        "avg_monthly_usage_gb": usage,
        "support_calls_6m": support,
        "late_payments_12m": late,
        "contract_type": contract,
        "internet_service": internet,
        "payment_method": payment,
        "senior_citizen": senior,
        "has_partner": partner,
        "paperless_billing": paperless,
    }])
    probability = float(model.predict_proba(row)[0, 1])
    st.metric("Estimated churn probability", f"{probability:.1%}")
    if probability >= 0.5:
        st.warning("This customer is above the current 50% review threshold.")
    else:
        st.success("This customer is below the current 50% review threshold.")
    st.caption("This demo is trained on synthetic data and should not be used for real customer decisions.")
