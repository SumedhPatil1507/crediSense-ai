import streamlit as st
import joblib
import pandas as pd
import json
from pathlib import Path

# 🔥 Import helper
from app.utils import build_full_input

# --- PATH SETUP ---
BASE_DIR = Path(__file__).resolve().parents[2]

model_path = BASE_DIR / "models" / "model.pkl"
columns_path = BASE_DIR / "models" / "columns.json"

# --- LOAD MODEL ---
model = joblib.load(model_path)

# --- LOAD COLUMNS ---
try:
    cols = json.load(open(columns_path))
except:
    st.error("❌ columns.json not found")
    st.stop()

# --- UI ---
st.title("🔍 Loan Risk Prediction")

st.markdown("### Enter Applicant Details")

# 🔥 INPUTS 
col1, col2 = st.columns(2)

with col1:
    income = st.slider("Income", 0.0, 1.0, 0.5)
    age = st.slider("Age", 0.0, 1.0, 0.3)

with col2:
    exp = st.slider("Experience", 0.0, 1.0, 0.2)

# --- PREDICT ---
if st.button("🚀 Predict Risk"):

    # Build user input
    user_input = {
        "Income": income,
        "Age": age,
        "Experience": exp
    }

    # 🔥 Build FULL dataframe
    df = build_full_input(user_input, cols)

    # Predict
    prob = model.predict_proba(df)[0][1]

    # --- OUTPUT ---
    st.markdown("---")

    col1, col2 = st.columns(2)

    col1.metric("Risk Probability", f"{prob:.2%}")

    decision = "Approve" if prob < 0.3 else "Review" if prob < 0.6 else "Reject"
    col2.metric("Decision", decision)

    if prob > 0.5:
        st.error("⚠️ High Risk Applicant")
    else:
        st.success("✅ Safe Applicant")

    # 💰 Business logic
    loan = 100000
    profit = (1 - prob)*20000 - prob*loan
    st.metric("Expected Profit", f"₹{profit:,.0f}")