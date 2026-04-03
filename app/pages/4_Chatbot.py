import streamlit as st
import joblib
import json
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from utils import build_full_input

model = joblib.load(BASE_DIR / "models/model.pkl")
cols = json.load(open(BASE_DIR / "models/columns.json"))

st.title("🤖 AI Risk Assistant")

st.write("Enter values like: 0.5,0.3,0.2")

text = st.text_input("Input:")

if text:
    try:
        vals = text.split(",")

        if len(vals) != 3:
            st.warning("Enter exactly 3 values")
            st.stop()

        income, age, exp = map(float, vals)

        user_input = {
            "Income": income,
            "Age": age,
            "Experience": exp
        }

        df = build_full_input(user_input, cols)

        prob = model.predict_proba(df)[0][1]

        if prob > 0.5:
            st.error(f"⚠️ High Risk ({prob:.2%})")
        else:
            st.success(f"✅ Safe ({prob:.2%})")

    except Exception as e:
        st.error("Invalid format. Use: 0.5,0.3,0.2")