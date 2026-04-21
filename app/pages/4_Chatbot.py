import streamlit as st
import joblib
import json
import warnings
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from utils import build_full_input
from src.feedback import log_prediction, log_feedback

model = joblib.load(BASE_DIR / "models/model.pkl")
with open(BASE_DIR / "models/columns.json") as f:
    cols = json.load(f)

st.set_page_config(layout="wide")
st.title("🤖 AI Risk Assistant")

tabs = st.tabs(["🔮 Risk Prediction", "💬 Credit Risk Q&A"])

# ── TAB 1: Prediction ─────────────────────────────────────────────────────────
with tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        income = st.slider("Income (normalized 0–1)", 0.0, 1.0, 0.5,
                           help="LPA ÷ 50. e.g. 10 LPA → 0.20")
        age    = st.slider("Age (normalized 0–1)", 0.0, 1.0, 0.3,
                           help="(age − 18) ÷ 52")
    with col2:
        exp    = st.slider("Experience (normalized 0–1)", 0.0, 1.0, 0.2,
                           help="Years ÷ 40")
        st.caption("**Guide:** Income = LPA/50 · Age = (age−18)/52 · Exp = years/40")

    if st.button("🚀 Assess Risk", use_container_width=True):
        user_input = {"Income": income, "Age": age, "Experience": exp}
        df_in = build_full_input(user_input, cols)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob = model.predict_proba(df_in)[0][1]

        safe_prob = 1 - prob
        decision = "Approve" if prob < 0.3 else "Manual Review" if prob < 0.6 else "Reject"
        margin = abs(prob - 0.5)
        confidence = "High" if margin > 0.3 else "Medium" if margin > 0.15 else "Low (borderline)"

        st.session_state["chatbot_pred"] = dict(income=income, age=age, exp=exp,
                                                 prob=prob, decision=decision)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Default Risk", f"{prob:.2%}")
        r2.metric("Safe Probability", f"{safe_prob:.2%}")
        r3.metric("Confidence", confidence)
        r4.metric("Decision", decision)

        if prob > 0.5:
            st.error(f"⚠️ High Default Risk: {prob:.2%}")
        elif prob > 0.3:
            st.warning(f"🔍 Moderate Risk: {prob:.2%}")
        else:
            st.success(f"✅ Low Risk: {prob:.2%}")

        # Explanation
        drivers = []
        if income < 0.3: drivers.append("below-average income")
        if exp < 0.2:    drivers.append("limited work experience")
        if age < 0.2:    drivers.append("young applicant profile")
        if income > 0.7 and exp > 0.5: drivers.append("strong income & experience")
        driver_text = f" Key factors: {', '.join(drivers)}." if drivers else ""
        risk_label = "low" if prob < 0.3 else "moderate" if prob < 0.6 else "high"
        st.info(f"This applicant has a **{risk_label}** default risk ({prob:.1%}).{driver_text} "
                f"Recommendation: **{decision}**.")

        log_prediction(income, age, exp, prob, decision, page="Chatbot")
        st.caption("📚 Model: LightGBM · [Kaggle Dataset](https://www.kaggle.com/datasets/subhamjain/loan-prediction-based-on-customer-behavior)")

    # Feedback
    if "chatbot_pred" in st.session_state:
        st.markdown("---")
        st.subheader("📣 Feedback")
        p = st.session_state["chatbot_pred"]
        feedback = st.radio("Was this prediction correct?",
                            ["👍 Correct", "👎 Incorrect", "🤔 Unsure"], horizontal=True)
        corrected = st.selectbox("Correct label:", ["", "Should be Approve",
                                                     "Should be Reject", "Should be Review"])
        notes = st.text_input("Notes:", placeholder="Optional context")
        if st.button("Submit Feedback"):
            log_feedback(p["income"], p["age"], p["exp"], p["prob"],
                         p["decision"], feedback, corrected, notes)
            st.success("✅ Feedback recorded!")

# ── TAB 2: Credit Risk Q&A ────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("💬 Credit Risk Knowledge Base")
    st.caption("Common questions about credit risk, model metrics, and lending decisions.")

    qa = {
        "What is credit risk?": (
            "Credit risk is the probability that a borrower defaults on a loan. "
            "It's assessed using income, employment stability, age, and repayment history. "
            "This model predicts default probability using LightGBM trained on 252,000 loan records."
        ),
        "What is the KS statistic?": (
            "The Kolmogorov-Smirnov (KS) statistic measures the maximum separation between "
            "the cumulative distributions of defaulters and non-defaulters. "
            "A KS > 0.4 is considered good for credit models. Industry standard metric."
        ),
        "What is the Gini coefficient?": (
            "Gini = 2 × AUC − 1. It measures the model's discriminatory power. "
            "A Gini of 0.7+ is considered strong. It's the primary metric used by banks and credit bureaus."
        ),
        "How does the decision threshold work?": (
            "The model outputs a probability (0–1). We apply thresholds: "
            "< 0.3 → Approve, 0.3–0.6 → Manual Review, > 0.6 → Reject. "
            "Lowering the threshold catches more defaults but increases false rejections."
        ),
        "What does confidence mean?": (
            "Confidence reflects how far the predicted probability is from the 0.5 decision boundary. "
            "High confidence (>0.3 margin) = clear case. Low confidence = borderline, needs manual review."
        ),
        "Why LightGBM over other models?": (
            "LightGBM uses gradient-based leaf-wise splitting which handles class imbalance better. "
            "It outperforms Logistic Regression and Random Forest on AUC, Gini, and KS on this dataset. "
            "It also trains faster on large tabular datasets."
        ),
        "What is PR-AUC?": (
            "Precision-Recall AUC is more informative than ROC-AUC for imbalanced datasets. "
            "Credit risk data is imbalanced (~12% defaults), so PR-AUC better reflects real performance. "
            "A PR-AUC of 0.6+ on imbalanced data is strong."
        ),
    }

    for question, answer in qa.items():
        with st.expander(f"❓ {question}"):
            st.write(answer)

    st.markdown("---")
    st.caption("📚 References: [Basel II Credit Risk Framework](https://www.bis.org/publ/bcbs128.htm) · "
               "[scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) · "
               "[LightGBM Paper (NeurIPS 2017)](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)")
