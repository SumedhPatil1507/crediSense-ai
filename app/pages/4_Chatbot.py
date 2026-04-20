import streamlit as st
import joblib
import json
import warnings
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from utils import build_full_input
from src.llm import explain_prediction, chat_with_analyst

model = joblib.load(BASE_DIR / "models/model.pkl")
with open(BASE_DIR / "models/columns.json") as f:
    cols = json.load(f)

st.set_page_config(layout="wide")
st.title("🤖 AI Risk Assistant")

tabs = st.tabs(["🔮 Risk Prediction", "💬 Ask the Analyst"])

# ── TAB 1: Prediction with explanation ────────────────────────────────────────
with tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        income = st.slider("Income (normalized 0–1)", 0.0, 1.0, 0.5)
        age = st.slider("Age (normalized 0–1)", 0.0, 1.0, 0.3)
    with col2:
        exp = st.slider("Experience (normalized 0–1)", 0.0, 1.0, 0.2)
        st.markdown("**Normalization guide:**")
        st.caption("Income: LPA / 50 · Age: (age-18)/52 · Experience: years/40")

    if st.button("🚀 Assess Risk", use_container_width=True):
        user_input = {"Income": income, "Age": age, "Experience": exp}
        df_in = build_full_input(user_input, cols)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob = model.predict_proba(df_in)[0][1]

        r1, r2, r3 = st.columns(3)
        r1.metric("Risk Probability", f"{prob:.2%}")
        r2.metric("Decision", "✅ Approve" if prob < 0.3 else "🔍 Review" if prob < 0.6 else "❌ Reject")
        r3.metric("Risk Level", "Low" if prob < 0.3 else "Medium" if prob < 0.6 else "High")

        if prob > 0.5:
            st.error(f"⚠️ High Default Risk: {prob:.2%}")
        elif prob > 0.3:
            st.warning(f"🔍 Moderate Risk: {prob:.2%}")
        else:
            st.success(f"✅ Low Risk: {prob:.2%}")

        st.markdown("---")
        st.subheader("📝 Explanation")
        st.info(explain_prediction(prob, income, age, exp))

# ── TAB 2: Q&A ────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("💬 Ask the Credit Risk Analyst")
    st.caption("Ask about credit risk concepts, model metrics, or lending decisions.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_q = st.chat_input("e.g. What is the KS statistic? How does threshold affect approvals?")

    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.write(user_q)

        reply = chat_with_analyst(st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.write(reply)

    if st.session_state.chat_history and st.button("🗑️ Clear chat"):
        st.session_state.chat_history = []
        st.rerun()
