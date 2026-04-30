import streamlit as st
import joblib
import json
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from utils import build_full_input
from src.data_loader import load_data
from src.preprocessing import clean_data
from src.feature_engineering import create_features
from src.evaluate import evaluate, threshold_analysis, compare_models
from src.database import log_prediction, log_feedback, log_audit, db_status
from src.confidence_intervals import bootstrap_ci, interpret_ci
from src.adverse_action import generate_adverse_action
from src.report import generate_pdf_report, FPDF_AVAILABLE
from src.config import THRESHOLD_APPROVE, THRESHOLD_REVIEW

model = joblib.load(BASE_DIR / "models/model.pkl")
with open(BASE_DIR / "models/columns.json") as f:
    cols = json.load(f)

@st.cache_data
def load_dataset():
    df = load_data(str(BASE_DIR / "data" / "loan_cleaned.csv"))
    df = clean_data(df)
    df = create_features(df)
    return df

def normalize(income_lpa, age_years, exp_years):
    return (min(income_lpa / 50.0, 1.0),
            (age_years - 18) / 52.0,
            min(exp_years / 40.0, 1.0))

st.set_page_config(layout="wide")
st.title("💳 Loan Risk Prediction")

# Onboarding
if "onboarded" not in st.session_state:
    with st.expander("👋 Quick Start Guide", expanded=True):
        st.markdown("""
        1. **Predict** — enter real applicant values (LPA, age, years) and get a risk score
        2. **What-If Simulator** — compare two scenarios side by side
        3. **Model Performance** — ROC-AUC, Gini, KS, calibration (auto-loaded)
        4. **Model Comparison** — LightGBM vs baselines
        5. **Threshold Analysis** — tune the decision cutoff
        """)
        if st.button("Got it"):
            st.session_state["onboarded"] = True
            st.rerun()

# DB status badge
db = db_status()
st.caption(f"DB: SQLite (persistent local database)")

tabs = st.tabs(["🔮 Predict", "🔬 What-If Simulator", "📈 Model Performance",
                "🏆 Model Comparison", "⚙️ Threshold Analysis"])

# ── TAB 1: Predict ─────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Applicant Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        income_lpa = st.number_input("Annual Income (LPA)", min_value=0.5, max_value=500.0,
                                      value=8.0, step=0.5, help="Lakhs Per Annum")
        profession = st.selectbox("Profession", ["Engineer", "Doctor", "Lawyer", "Teacher",
                                                   "Accountant", "Manager", "Analyst", "Other"])
    with c2:
        age_years = st.number_input("Age (years)", min_value=18, max_value=70, value=30)
        house_own = st.selectbox("House Ownership", ["owned", "rented", "norent_noown"])
    with c3:
        exp_years = st.number_input("Work Experience (years)", min_value=0, max_value=45, value=5)
        marital   = st.selectbox("Marital Status", ["single", "married"])

    c4, c5, c6 = st.columns(3)
    with c4:
        car_own = st.selectbox("Car Ownership", ["no", "yes"])
    with c5:
        job_yrs  = st.number_input("Years in Current Job", min_value=0, max_value=40, value=2)
    with c6:
        house_yrs = st.number_input("Years in Current House", min_value=0, max_value=40, value=3)

    if st.button("🚀 Assess Risk", use_container_width=True):
        if exp_years >= age_years - 16:
            st.error("Experience cannot exceed working age.")
            st.stop()

        income_n, age_n, exp_n = normalize(income_lpa, age_years, exp_years)
        age_group = "Young" if age_n < 0.3 else "Senior" if age_n > 0.7 else "Middle"

        user_input = {
            "Income": income_n, "Age": age_n, "Experience": exp_n,
            "CURRENT_JOB_YRS": job_yrs, "CURRENT_HOUSE_YRS": house_yrs,
            "House_Ownership": house_own, "Married/Single": marital,
            "Car_Ownership": car_own, "Profession": profession,
            "CITY": "Mumbai", "STATE": "Maharashtra", "age_group": age_group,
        }
        df_input = build_full_input(user_input, cols)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob = float(model.predict_proba(df_input)[0][1])

        with st.spinner("Computing confidence interval..."):
            _, ci_lower, ci_upper = bootstrap_ci(model, df_input, n_bootstrap=150)

        decision   = "Approve" if prob < THRESHOLD_APPROVE else "Manual Review" if prob < THRESHOLD_REVIEW else "Reject"
        margin     = abs(prob - 0.5)
        confidence = "High" if margin > 0.3 else "Medium" if margin > 0.15 else "Low (borderline)"
        ci_note    = interpret_ci(ci_lower, ci_upper)

        st.session_state["last_pred"] = dict(
            income_lpa=income_lpa, age_years=age_years, exp_years=exp_years,
            income_n=income_n, age_n=age_n, exp_n=exp_n,
            prob=prob, decision=decision, confidence=confidence,
            ci_lower=ci_lower, ci_upper=ci_upper
        )

        # KPIs
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Default Risk",      f"{prob:.2%}")
        k2.metric("Safe Probability",  f"{1-prob:.2%}")
        k3.metric("95% CI",            f"[{ci_lower:.1%}, {ci_upper:.1%}]")
        k4.metric("Confidence",        confidence)
        k5.metric("Decision",          decision)
        st.caption(f"CI: {ci_note}")

        # Gauge
        conf_color = "#28a745" if prob < THRESHOLD_APPROVE else "#ffc107" if prob < THRESHOLD_REVIEW else "#dc3545"
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": conf_color},
                "steps": [
                    {"range": [0, 30],  "color": "#d4edda"},
                    {"range": [30, 60], "color": "#fff3cd"},
                    {"range": [60, 100],"color": "#f8d7da"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "value": 50}
            },
            title={"text": f"Risk Score | 95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]"}
        ))
        st.plotly_chart(gauge, use_container_width=True)

        if decision == "Approve":   st.success(f"✅ Decision: {decision}")
        elif decision == "Manual Review": st.warning(f"🔍 Decision: {decision}")
        else:                        st.error(f"❌ Decision: {decision}")

        # Explanation
        drivers = []
        if income_lpa < 5:   drivers.append(f"low income ({income_lpa} LPA)")
        if exp_years < 2:    drivers.append("limited work experience")
        if age_years < 25:   drivers.append("young applicant")
        if income_lpa > 20 and exp_years > 8: drivers.append("strong income and experience")
        driver_text = f" Key factors: {', '.join(drivers)}." if drivers else ""
        risk_label  = "low" if prob < THRESHOLD_APPROVE else "moderate" if prob < THRESHOLD_REVIEW else "high"
        explanation = f"This applicant has a {risk_label} default risk ({prob:.1%}).{driver_text} Recommendation: {decision}."
        st.info(explanation)

        # Adverse action
        adverse = generate_adverse_action(prob, income_n, age_n, exp_n)
        if adverse["required"]:
            st.markdown("---")
            st.subheader("Adverse Action Notice")
            st.warning(adverse["notice"])
            st.caption(adverse["citation"])

        # PDF
        if FPDF_AVAILABLE:
            pdf_bytes = generate_pdf_report(income_n, age_n, exp_n, prob, decision,
                                             confidence, ci_lower, ci_upper, explanation, adverse)
            if pdf_bytes:
                st.download_button("📄 Download PDF Report", pdf_bytes,
                                   "credisense_report.pdf", "application/pdf")

        # Log
        from src.validation import hash_input
        pred_id = log_prediction(income_lpa, age_years, exp_years,
                                  income_n, age_n, exp_n, prob, ci_lower, ci_upper,
                                  decision, confidence, page="Streamlit")
        log_audit("PREDICT", hash_input(income_n, age_n, exp_n),
                  details=f"decision={decision} prob={prob:.4f}")

        st.caption("Dataset: [Kaggle](https://www.kaggle.com/datasets/subhamjain/loan-prediction-based-on-customer-behavior) | "
                   "Model: [LightGBM NeurIPS 2017](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)")

    # Feedback
    if "last_pred" in st.session_state:
        st.markdown("---")
        st.subheader("Analyst Feedback")
        p = st.session_state["last_pred"]
        fb_c1, fb_c2 = st.columns(2)
        with fb_c1:
            feedback = st.radio("Was this prediction correct?",
                                ["correct", "incorrect", "unsure"],
                                horizontal=True, key="model_fb_radio")
        with fb_c2:
            corrected = st.selectbox("Correct label:", ["", "Should be Approve",
                                                         "Should be Reject", "Should be Review"],
                                     key="model_fb_select")
        notes = st.text_input("Notes:", placeholder="Optional context", key="model_fb_notes")
        if st.button("Submit Feedback", key="model_fb_submit"):
            pred_id = st.session_state.get("last_pred_id", "unknown")
            log_feedback(pred_id, feedback, corrected, notes)
            st.success("Feedback recorded.")

# ── TAB 2: What-If Simulator ───────────────────────────────────────────────────
with tabs[1]:
    st.subheader("What-If Scenario Simulator")
    st.caption("Compare two applicant profiles side by side.")

    wc1, wc2 = st.columns(2)
    scenarios = {}
    for label, col in [("Baseline", wc1), ("Scenario", wc2)]:
        with col:
            st.markdown(f"**{label}**")
            inc = st.number_input(f"Income LPA ({label})", 0.5, 500.0, 6.0 if label=="Baseline" else 15.0, key=f"wi_{label}_inc")
            ag  = st.number_input(f"Age ({label})", 18, 70, 28 if label=="Baseline" else 35, key=f"wi_{label}_age")
            ex  = st.number_input(f"Experience ({label})", 0, 45, 3 if label=="Baseline" else 10, key=f"wi_{label}_exp")
            scenarios[label] = (inc, ag, ex)

    if st.button("Compare"):
        results = []
        for label, (inc, ag, ex) in scenarios.items():
            inc_n, age_n, exp_n = normalize(inc, ag, ex)
            df_w = build_full_input({"Income": inc_n, "Age": age_n, "Experience": exp_n}, cols)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = float(model.predict_proba(df_w)[0][1])
            dec = "Approve" if p < THRESHOLD_APPROVE else "Review" if p < THRESHOLD_REVIEW else "Reject"
            results.append({"Scenario": label, "Income (LPA)": inc, "Age": ag,
                             "Experience (yrs)": ex, "Risk %": f"{p:.2%}", "Decision": dec, "_p": p})

        df_res = pd.DataFrame(results)
        st.dataframe(df_res.drop(columns=["_p"]), use_container_width=True)
        delta = results[1]["_p"] - results[0]["_p"]
        if delta < 0:   st.success(f"Scenario reduces risk by {abs(delta):.2%}")
        elif delta > 0: st.error(f"Scenario increases risk by {delta:.2%}")
        else:           st.info("No change in risk.")

        fig = go.Figure(go.Bar(
            x=["Baseline", "Scenario"],
            y=[results[0]["_p"], results[1]["_p"]],
            marker_color=["royalblue", "darkorange"],
            text=[f"{results[0]['_p']:.2%}", f"{results[1]['_p']:.2%}"],
            textposition="outside"
        ))
        fig.update_layout(yaxis_title="Default Probability", yaxis_range=[0, 1], height=350)
        st.plotly_chart(fig, use_container_width=True)

# ── TAB 3: Model Performance ───────────────────────────────────────────────────
with tabs[2]:
    st.caption("Auto-loaded from repo.")
    with st.spinner("Evaluating..."):
        try:
            from sklearn.model_selection import train_test_split
            df_eval = load_dataset()
            X_eval = df_eval.drop(columns=["Risk_Flag"])
            y_eval = df_eval["Risk_Flag"]
            _, X_test, _, y_test = train_test_split(X_eval, y_eval, test_size=0.2, stratify=y_eval, random_state=42)
            m = evaluate(model, X_test, y_test)

            k1,k2,k3,k4,k5 = st.columns(5)
            k1.metric("ROC-AUC", m["AUC"])
            k2.metric("PR-AUC",  m["PR_AUC"])
            k3.metric("Gini",    m["Gini"])
            k4.metric("KS Stat", m["KS"])
            k5.metric("Brier",   m["Brier"])
            st.caption("KS & Gini are standard credit-scoring metrics. Lower Brier = better calibration.")
            st.markdown("---")

            st.subheader("Classification Report")
            report_df = pd.DataFrame(m["report"]).T.drop(columns=["support"], errors="ignore")
            st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
            st.markdown("---")

            c_roc, c_cm = st.columns(2)
            with c_roc:
                fpr, tpr, _ = m["roc_curve"]
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"LightGBM AUC={m['AUC']}", line=dict(color="royalblue", width=2)))
                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random", line=dict(dash="dash", color="gray")))
                fig_roc.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR", height=350)
                st.plotly_chart(fig_roc, use_container_width=True)

            with c_cm:
                cm = np.array(m["confusion_matrix"])
                fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"),
                                   x=["Safe","Risk"], y=["Safe","Risk"],
                                   color_continuous_scale="Blues", title="Confusion Matrix")
                fig_cm.update_layout(height=350)
                st.plotly_chart(fig_cm, use_container_width=True)

            c_pr, c_cal = st.columns(2)
            with c_pr:
                prec, rec, _ = m["pr_curve"]
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=rec, y=prec, name=f"PR-AUC={m['PR_AUC']}", line=dict(color="darkorange", width=2)))
                fig_pr.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision", height=350)
                st.plotly_chart(fig_pr, use_container_width=True)

            with c_cal:
                frac_pos, mean_pred = m["calibration"]
                fig_cal = go.Figure()
                fig_cal.add_trace(go.Scatter(x=mean_pred, y=frac_pos, name="Model", mode="lines+markers", line=dict(color="green")))
                fig_cal.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Perfect", line=dict(dash="dash", color="gray")))
                fig_cal.update_layout(title="Calibration Curve", xaxis_title="Mean Predicted Prob", yaxis_title="Fraction Positives", height=350)
                st.plotly_chart(fig_cal, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")

# ── TAB 4: Model Comparison ────────────────────────────────────────────────────
with tabs[3]:
    if st.button("Run Model Comparison"):
        with st.spinner("Training baselines (~30s)..."):
            try:
                from sklearn.model_selection import train_test_split
                from sklearn.linear_model import LogisticRegression
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.pipeline import Pipeline
                from src.encoding import build_preprocessor
                df_c = load_dataset()
                X_c = df_c.drop(columns=["Risk_Flag"]); y_c = df_c["Risk_Flag"]
                X_tr, X_te, y_tr, y_te = train_test_split(X_c, y_c, test_size=0.2, stratify=y_c, random_state=42)
                lr = Pipeline([("pre", build_preprocessor(X_tr)), ("m", LogisticRegression(class_weight="balanced", max_iter=300, random_state=42))])
                rf = Pipeline([("pre", build_preprocessor(X_tr)), ("m", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1))])
                lr.fit(X_tr, y_tr); rf.fit(X_tr, y_tr)
                results = compare_models({"Logistic Regression": lr, "Random Forest": rf, "LightGBM": model}, X_te, y_te)
                df_comp = pd.DataFrame(results)
                st.dataframe(df_comp.style.highlight_max(subset=["ROC-AUC","PR-AUC","Gini","KS","F1"], color="#d4edda").highlight_min(subset=["Brier"], color="#d4edda").format("{:.4f}", subset=df_comp.columns[1:]), use_container_width=True)
                fig_bar = px.bar(df_comp, x="Model", y="ROC-AUC", color="Model", title="ROC-AUC Comparison", text_auto=".4f")
                st.plotly_chart(fig_bar, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

# ── TAB 5: Threshold Analysis ──────────────────────────────────────────────────
with tabs[4]:
    with st.spinner("Running threshold analysis..."):
        try:
            from sklearn.model_selection import train_test_split
            df_t = load_dataset()
            X_t = df_t.drop(columns=["Risk_Flag"]); y_t = df_t["Risk_Flag"]
            _, X_te_t, _, y_te_t = train_test_split(X_t, y_t, test_size=0.2, stratify=y_t, random_state=42)
            y_prob_t = model.predict_proba(X_te_t)[:, 1]
            results = threshold_analysis(y_te_t, y_prob_t)
            df_thresh = pd.DataFrame(results)
            fig_thresh = go.Figure()
            for col_name, color in [("precision","blue"),("recall","red"),("f1","green"),("approval_rate","orange")]:
                fig_thresh.add_trace(go.Scatter(x=df_thresh["threshold"], y=df_thresh[col_name], name=col_name.capitalize(), line=dict(color=color)))
            fig_thresh.update_layout(xaxis_title="Threshold", yaxis_title="Score", height=400, title="Threshold vs Business Metrics")
            st.plotly_chart(fig_thresh, use_container_width=True)
            st.dataframe(df_thresh.style.format("{:.3f}"), use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
