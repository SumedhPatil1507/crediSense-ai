import streamlit as st
import joblib
import json
import warnings
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from utils import build_full_input
from src.llm import explain_prediction, parse_natural_language_input, chat_with_analyst, get_groq_client

model = joblib.load(BASE_DIR / "models/model.pkl")
with open(BASE_DIR / "models/columns.json") as f:
    cols = json.load(f)

st.set_page_config(layout="wide")
st.title("🤖 AI Risk Assistant")

llm_available = get_groq_client() is not None
if llm_available:
    st.success("🟢 LLM connected (Groq · llama3-8b)")
else:
    st.warning("🟡 LLM not configured — using rule-based fallback. Add GROQ_API_KEY to Streamlit secrets for full AI explanations.")

tabs = st.tabs(["🔮 Risk Prediction", "💬 Ask the Analyst"])

# ── TAB 1: Prediction with LLM explanation ────────────────────────────────────
with tabs[0]:
    st.markdown("Describe the applicant in plain English **or** enter normalized values.")

    col_l, col_r = st.columns([3, 2])

    with col_l:
        nl_input = st.text_area(
            "Natural language input (LLM-powered):",
            placeholder="e.g. 32 year old software engineer in Bangalore, earning 12 LPA, 6 years experience",
            height=80
        )
        st.caption("— or use manual sliders —")
        income = st.slider("Income (normalized 0–1)", 0.0, 1.0, 0.5)
        age = st.slider("Age (normalized 0–1)", 0.0, 1.0, 0.3)
        exp = st.slider("Experience (normalized 0–1)", 0.0, 1.0, 0.2)

    with col_r:
        st.markdown("**Normalization guide:**")
        st.markdown("""
        | Field | Formula |
        |-------|---------|
        | Income | LPA / 50 |
        | Age | (age - 18) / 52 |
        | Experience | years / 40 |
        """)

    if st.button("🚀 Assess Risk", use_container_width=True):
        # Try NL parsing first
        parsed = None
        if nl_input.strip() and llm_available:
            with st.spinner("Parsing natural language input..."):
                parsed = parse_natural_language_input(nl_input)
            if parsed:
                income = parsed.get("Income", income)
                age = parsed.get("Age", age)
                exp = parsed.get("Experience", exp)
                st.info(f"Parsed → Income: {income:.3f}, Age: {age:.3f}, Experience: {exp:.3f}")

        user_input = {"Income": income, "Age": age, "Experience": exp}
        df_in = build_full_input(user_input, cols)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob = model.predict_proba(df_in)[0][1]

        # Display result
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Risk Probability", f"{prob:.2%}")
        decision = "✅ Approve" if prob < 0.3 else "🔍 Manual Review" if prob < 0.6 else "❌ Reject"
        res_col2.metric("Decision", decision)
        res_col3.metric("Risk Level", "Low" if prob < 0.3 else "Medium" if prob < 0.6 else "High")

        if prob > 0.5:
            st.error(f"⚠️ High Default Risk: {prob:.2%}")
        elif prob > 0.3:
            st.warning(f"🔍 Moderate Risk: {prob:.2%}")
        else:
            st.success(f"✅ Low Risk: {prob:.2%}")

        # LLM explanation
        st.markdown("---")
        st.subheader("📝 AI Explanation")
        with st.spinner("Generating explanation..."):
            explanation = explain_prediction(prob, income, age, exp)
        st.info(explanation)

# ── TAB 2: Conversational Q&A ─────────────────────────────────────────────────
with tabs[1]:
    st.subheader("💬 Ask the Credit Risk Analyst")
    st.caption("Ask anything about credit risk, model decisions, or lending strategy.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_q = st.chat_input("Ask a question about credit risk...")

    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.write(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = chat_with_analyst(st.session_state.chat_history)
            st.write(reply)

        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    if st.button("🗑️ Clear chat"):
        st.session_state.chat_history = []
        st.rerun()
