import streamlit as st
import joblib
import pandas as pd
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from utils import build_full_input 

# Load model + columns
model = joblib.load(BASE_DIR / "models/model.pkl")
cols = json.load(open(BASE_DIR / "models/columns.json"))

st.title("🔍 Loan Risk Prediction")

col1, col2 = st.columns(2)

with col1:
    income = st.slider("Income", 0.0, 1.0, 0.5)
    age = st.slider("Age", 0.0, 1.0, 0.3)

with col2:
    exp = st.slider("Experience", 0.0, 1.0, 0.2)

if st.button("🚀 Predict Risk"):

    user_input = {
        "Income": income,
        "Age": age,
        "Experience": exp
    }

    df = build_full_input(user_input, cols)

    prob = model.predict_proba(df)[0][1]

    st.metric("Risk Probability", f"{prob:.2%}")

    decision = "Approve" if prob < 0.3 else "Review" if prob < 0.6 else "Reject"
    st.write(f"Decision: {decision}")

    if prob > 0.5:
        st.error("⚠️ High Risk")
    else:
        st.success("✅ Safe")