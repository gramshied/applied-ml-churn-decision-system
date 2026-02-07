import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Churn Decision System", layout="centered")

st.title("📊 Customer Churn Decision System")
st.write("Predict churn risk and understand key drivers behind the prediction.")

# Load model
model = joblib.load("src/final_model.joblib")

# Load a template row to preserve schema
X_train = pd.read_csv("data/X_train.csv")
input_data = X_train.iloc[[0]].copy()

# User-friendly inputs (overwrite selected fields)
input_data["tenure"] = st.slider("Tenure (months)", 0, 72, int(input_data["tenure"].iloc[0]))
input_data["MonthlyCharges"] = st.slider(
    "Monthly Charges", 0.0, 150.0, float(input_data["MonthlyCharges"].iloc[0])
)
input_data["Contract"] = st.selectbox(
    "Contract Type", ["Month-to-month", "One year", "Two year"]
)
input_data["InternetService"] = st.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"]
)

if st.button("Predict Churn"):
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("🔍 Prediction Result")
    st.metric("Churn Probability", f"{probability:.2%}")

    if probability > 0.5:
        st.error("⚠️ High risk of churn. Retention action recommended.")
    else:
        st.success("✅ Low churn risk.")
