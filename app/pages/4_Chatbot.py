import streamlit as st
import joblib
import json
import warnings
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from utils import build_full_input

model = joblib.load(BASE_DIR / "models/model.pkl")
with open(BASE_DIR / "models/columns.json") as f:
    cols = json.load(f)

st.title("🤖 AI Risk Assistant")

st.write("Enter normalized values (0–1) for: **Income, Age, Experience**")
st.caption("Example: `0.5, 0.3, 0.2`")

text = st.text_input("Input (Income, Age, Experience):")

if text:
    try:
        parts = [v.strip() for v in text.split(",")]

        if len(parts) != 3:
            st.warning("Please enter exactly 3 comma-separated values: Income, Age, Experience")
            st.stop()

        income, age, exp = map(float, parts)

        user_input = {
            "Income": income,
            "Age": age,
            "Experience": exp
        }

        df = build_full_input(user_input, cols)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob = model.predict_proba(df)[0][1]

        st.markdown(f"**Risk Probability:** `{prob:.2%}`")

        if prob > 0.5:
            st.error(f"⚠️ High Risk ({prob:.2%})")
        else:
            st.success(f"✅ Low Risk ({prob:.2%})")

    except ValueError as e:
        st.error(f"Invalid input: {e}. Make sure all values are numbers (e.g. 0.5, 0.3, 0.2)")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
