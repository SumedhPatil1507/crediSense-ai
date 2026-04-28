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
                                      value=8.0, step=0.5)
        profession = st.selectbox("Profession", ["Engineer", "Doctor", "Lawyer", "Teacher",
                                                   "Accountant", "Manager", "Analyst", "Other"])
    with c2:
        age_years = st.number_input("Age (years)", min_value=18, max_value=70, value=30)
        house_own = st.selectbox("House Ownership", ["owned", "rented", "norent_noown"])
    with c3:
        exp_years = st.number_input("Work Experience (years)", min_value=0, max_value=45, value=5)
        marital   = st.selectbox("Marital Status", ["single", "married"])

    if st.button("Assess Risk", use_container_width=True, type="primary"):
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

        # Auto-queue borderline cases
        if decision == "Manual Review" or confidence == "Low (borderline)":
            pred_id = log_prediction(income_lpa, age_years, exp_years,
                                      income_n, age_n, exp_n, prob, ci_lower, ci_upper,
                                      decision, confidence, page="Chatbot")
            hitl_enqueue(pred_id, income_lpa, age_years, exp_years, prob, ci_lower, ci_upper,
                         reason="Borderline case from Chatbot")
            st.info("This case has been added to the analyst review queue.")
        else:
            log_prediction(income_lpa, age_years, exp_years,
                           income_n, age_n, exp_n, prob, ci_lower, ci_upper,
                           decision, confidence, page="Chatbot")

    # Feedback
    if "chatbot_pred" in st.session_state:
        st.markdown("---")
        p = st.session_state["chatbot_pred"]
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
            "Gini = 2 x AUC - 1. It measures the model's discriminatory power. "
            "A Gini of 0.7+ is considered strong. Primary metric used by banks and credit bureaus."
        ),
        "How does the decision threshold work?": (
            "The model outputs a probability (0-1). Thresholds: "
            "< 30% = Approve, 30-60% = Manual Review, > 60% = Reject. "
            "Lowering the threshold catches more defaults but increases false rejections."
        ),
        "What does confidence mean?": (
            "Confidence reflects how far the predicted probability is from the 0.5 boundary. "
            "High confidence (>0.3 margin) = clear case. Low = borderline, needs manual review."
        ),
        "Why LightGBM over other models?": (
            "LightGBM uses gradient-based leaf-wise splitting which handles class imbalance better. "
            "It outperforms Logistic Regression and Random Forest on AUC, Gini, and KS. "
            "It also trains faster on large tabular datasets."
        ),
        "What is PSI and why does it matter?": (
            "Population Stability Index (PSI) measures how much the score distribution has shifted "
            "between training and live data. PSI < 0.1 = stable, 0.1-0.2 = monitor, > 0.2 = retrain. "
            "It's the primary model monitoring metric in production credit systems."
        ),
        "What is an adverse action notice?": (
            "An adverse action notice is a legally required document (ECOA/FCRA) that explains "
            "why a loan application was declined. It must list specific reasons. "
            "CrediSense generates these automatically using SHAP-based feature attribution."
        ),
    }

    for question, answer in qa.items():
        with st.expander(f"{question}"):
            st.write(answer)

    st.markdown("---")
    st.caption("References: Basel II Credit Risk Framework (bis.org) | "
               "scikit-learn Metrics (scikit-learn.org) | "
               "LightGBM Paper NeurIPS 2017 | SHAP Paper NeurIPS 2017")

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
