import streamlit as st
import joblib
import json
from pathlib import Path

from utils import build_full_input   

BASE_DIR = Path(__file__).resolve().parents[2]

model = joblib.load(BASE_DIR / "models/model.pkl")
cols = json.load(open(BASE_DIR / "models/columns.json"))

st.title("🤖 Credit Risk Chatbot")

text = st.text_input("Enter: income,age,experience")

if text:
    try:
        vals = list(map(float, text.split(',')))

        user_input = {
            "Income": vals[0],
            "Age": vals[1],
            "Experience": vals[2]
        }

        df = build_full_input(user_input, cols)

        prob = model.predict_proba(df)[0][1]

        if prob > 0.5:
            st.error(f"⚠️ High Risk ({prob:.2%})")
        else:
            st.success(f"✅ Safe ({prob:.2%})")

    except:
        st.warning("Enter values like: 0.5,0.3,0.2")