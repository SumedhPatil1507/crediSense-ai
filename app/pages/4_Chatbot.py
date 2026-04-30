import streamlit as st
import joblib
import json
import warnings
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from utils import build_full_input
from src.database import log_prediction, log_feedback
from src.confidence_intervals import bootstrap_ci
from src.adverse_action import generate_adverse_action
from src.hitl_queue import enqueue as hitl_enqueue
from src.config import THRESHOLD_APPROVE, THRESHOLD_REVIEW

model = joblib.load(BASE_DIR / "models/model.pkl")
with open(BASE_DIR / "models/columns.json") as f:
    cols = json.load(f)

def normalize(income_lpa, age_years, exp_years):
    return (min(income_lpa / 50.0, 1.0),
            (age_years - 18) / 52.0,
            min(exp_years / 40.0, 1.0))

st.set_page_config(layout="wide")
st.title("AI Risk Assistant")

tabs = st.tabs(["Risk Prediction", "Credit Risk Q&A"])

# ── TAB 1: Prediction ─────────────────────────────────────────────────────────
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    with c1:
        income_lpa = st.number_input("Annual Income (LPA)", min_value=0.5, max_value=500.0,
                                      value=8.0, step=0.5, key="cb_income")
        profession = st.selectbox("Profession", ["Engineer", "Doctor", "Lawyer", "Teacher",
                                                   "Accountant", "Manager", "Analyst", "Other"],
                                  key="cb_profession")
    with c2:
        age_years = st.number_input("Age (years)", min_value=18, max_value=70, value=30, key="cb_age")
        house_own = st.selectbox("House Ownership", ["owned", "rented", "norent_noown"], key="cb_house")
    with c3:
        exp_years = st.number_input("Work Experience (years)", min_value=0, max_value=45,
                                     value=5, key="cb_exp")
        marital   = st.selectbox("Marital Status", ["single", "married"], key="cb_marital")

    if st.button("Assess Risk", use_container_width=True, type="primary", key="cb_predict"):
        if exp_years >= age_years - 16:
            st.error("Experience cannot exceed working age.")
            st.stop()

        income_n, age_n, exp_n = normalize(income_lpa, age_years, exp_years)
        age_group = "Young" if age_n < 0.3 else "Senior" if age_n > 0.7 else "Middle"

        user_input = {
            "Income": income_n, "Age": age_n, "Experience": exp_n,
            "CURRENT_JOB_YRS": 2, "CURRENT_HOUSE_YRS": 3,
            "House_Ownership": house_own, "Married/Single": marital,
            "Car_Ownership": "no", "Profession": profession,
            "CITY": "Mumbai", "STATE": "Maharashtra", "age_group": age_group,
        }
        df_in = build_full_input(user_input, cols)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob = float(model.predict_proba(df_in)[0][1])

        with st.spinner("Computing confidence interval..."):
            _, ci_lower, ci_upper = bootstrap_ci(model, df_in, n_bootstrap=100)

        decision   = "Approve" if prob < THRESHOLD_APPROVE else "Manual Review" if prob < THRESHOLD_REVIEW else "Reject"
        margin     = abs(prob - 0.5)
        confidence = "High" if margin > 0.3 else "Medium" if margin > 0.15 else "Low (borderline)"

        st.session_state["chatbot_pred"] = dict(
            income_lpa=income_lpa, age_years=age_years, exp_years=exp_years,
            income_n=income_n, age_n=age_n, exp_n=exp_n,
            prob=prob, decision=decision, confidence=confidence
        )

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Default Risk",     f"{prob:.2%}")
        r2.metric("Safe Probability", f"{1-prob:.2%}")
        r3.metric("Confidence",       confidence)
        r4.metric("Decision",         decision)
        st.caption(f"95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]")

        if prob > THRESHOLD_REVIEW:
            st.error(f"High Default Risk: {prob:.2%}")
        elif prob > THRESHOLD_APPROVE:
            st.warning(f"Moderate Risk: {prob:.2%}")
        else:
            st.success(f"Low Risk: {prob:.2%}")

        # Explanation
        drivers = []
        if income_lpa < 5:   drivers.append(f"low income ({income_lpa} LPA)")
        if exp_years < 2:    drivers.append("limited work experience")
        if age_years < 25:   drivers.append("young applicant")
        if income_lpa > 20 and exp_years > 8: drivers.append("strong income and experience")
        driver_text = f" Key factors: {', '.join(drivers)}." if drivers else ""
        risk_label  = "low" if prob < THRESHOLD_APPROVE else "moderate" if prob < THRESHOLD_REVIEW else "high"
        st.info(f"This applicant has a {risk_label} default risk ({prob:.1%}).{driver_text} Recommendation: {decision}.")

        # Adverse action
        adverse = generate_adverse_action(prob, income_n, age_n, exp_n)
        if adverse["required"]:
            st.warning(adverse["notice"])
            st.caption(adverse["citation"])

        # Log + auto-queue borderline cases
        pred_id = log_prediction(income_lpa, age_years, exp_years,
                                  income_n, age_n, exp_n, prob, ci_lower, ci_upper,
                                  decision, confidence, page="Chatbot")
        if decision == "Manual Review" or confidence == "Low (borderline)":
            hitl_enqueue(pred_id, income_lpa, age_years, exp_years, prob, ci_lower, ci_upper,
                         reason="Borderline case from Chatbot")
            st.info("This case has been added to the analyst review queue.")

    # Feedback
    if "chatbot_pred" in st.session_state:
        st.markdown("---")
        fb_c1, fb_c2 = st.columns(2)
        with fb_c1:
            feedback = st.radio("Was this prediction correct?",
                                ["correct", "incorrect", "unsure"],
                                horizontal=True, key="chatbot_fb_radio")
        with fb_c2:
            corrected = st.selectbox("Correct label:", ["", "Should be Approve",
                                                         "Should be Reject", "Should be Review"],
                                     key="chatbot_fb_select")
        notes = st.text_input("Notes:", placeholder="Optional context", key="chatbot_fb_notes")
        if st.button("Submit Feedback", key="chatbot_fb_submit"):
            log_feedback("chatbot", feedback, corrected, notes)
            st.success("Feedback recorded.")

# ── TAB 2: Credit Risk Q&A ────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Credit Risk Knowledge Base")

    qa = {
        "What is credit risk?": (
            "Credit risk is the probability that a borrower defaults on a loan. "
            "Assessed using income, employment stability, age, and repayment history. "
            "This model predicts default probability using LightGBM on 252,000 loan records."
        ),
        "What is the KS statistic?": (
            "KS measures the maximum separation between cumulative distributions of defaulters "
            "and non-defaulters. KS > 0.4 is considered good. Industry standard in credit scoring."
        ),
        "What is the Gini coefficient?": (
            "Gini = 2 x AUC - 1. Measures discriminatory power. "
            "Gini > 0.7 is strong. Primary metric used by banks and credit bureaus."
        ),
        "How does the decision threshold work?": (
            "< 30% = Approve, 30-60% = Manual Review, > 60% = Reject. "
            "Lower threshold catches more defaults but increases false rejections."
        ),
        "What does confidence mean?": (
            "Distance from the 0.5 boundary. High (>0.3 margin) = clear case. "
            "Low = borderline, auto-queued for human review."
        ),
        "Why LightGBM?": (
            "Gradient-based leaf-wise splitting handles class imbalance better. "
            "Outperforms Logistic Regression and Random Forest on AUC, Gini, KS."
        ),
        "What is PSI?": (
            "Population Stability Index measures score distribution shift. "
            "PSI < 0.1 = stable, 0.1-0.2 = monitor, > 0.2 = retrain."
        ),
        "What is an adverse action notice?": (
            "Legally required document (ECOA/FCRA) explaining why a loan was declined. "
            "CrediSense generates these automatically using SHAP feature attribution."
        ),
    }

    for question, answer in qa.items():
        with st.expander(question):
            st.write(answer)

    st.markdown("---")
    st.caption("References: Basel II (bis.org) | scikit-learn | LightGBM NeurIPS 2017 | SHAP NeurIPS 2017")
