from app.utils import build_full_input
from pathlib import Path
import json
import joblib
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[2]

model = joblib.load(BASE_DIR / "models/model.pkl")
cols = json.load(open(BASE_DIR / "models/columns.json"))

text = st.text_input("Enter: income,age,experience")

if text:
    vals = list(map(float, text.split(',')))

    user_input = {
        "Income": vals[0],
        "Age": vals[1],
        "Experience": vals[2]
    }

    df = build_full_input(user_input, cols)

    prob = model.predict_proba(df)[0][1]

    if prob > 0.5:
        st.error("⚠️ High Risk")
    else:
        st.success("✅ Safe")