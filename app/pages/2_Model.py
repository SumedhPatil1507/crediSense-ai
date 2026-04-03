import streamlit as st
import joblib
import json
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from utils import build_full_input

model = joblib.load(BASE_DIR / "models/model.pkl")
with open(BASE_DIR / "models/columns.json") as f:
    cols = json.load(f)

st.set_page_config(layout="wide")
st.title("💳 Loan Risk Prediction")

col1, col2 = st.columns(2)

with col1:
    income = st.slider("Income", 0.0, 1.0, 0.5)
    age = st.slider("Age", 0.0, 1.0, 0.3)

with col2:
    exp = st.slider("Experience", 0.0, 1.0, 0.2)

if st.button("🚀 Predict"):

    user_input = {
        "Income": income,
        "Age": age,
        "Experience": exp
    }

    df = build_full_input(user_input, cols)

    prob = model.predict_proba(df)[0][1]

    st.metric("Risk Probability", f"{prob:.2%}")

    if prob > 0.5:
        st.error("⚠️ High Risk")
    else:
        st.success("✅ Safe")

    decision = "Approve" if prob < 0.3 else "Review" if prob < 0.6 else "Reject"
    st.info(f"Decision: {decision}")