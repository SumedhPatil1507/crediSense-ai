import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.database import get_predictions, get_feedback, get_audit, db_status

st.set_page_config(layout="wide")
st.title("📋 Logs, Audit & Analytics")

db = db_status()
db_path_str = db.get('db_path', 'credisense.db')
st.caption(f"Storage: 🟢 Premium Local SQLite (Zero-Config, fully persistent database at `{db_path_str}`)")

tabs = st.tabs(["📊 Usage Logs", "📣 Feedback", "🔐 Audit Log",
                "💰 Cost-Benefit Tracker", "📈 Performance Monitor"])

# ── TAB 1: Usage Logs ─────────────────────────────────────────────────────────
with tabs[0]:
    usage = get_predictions(limit=500)
    if not usage:
        st.info("No predictions logged yet. Make a prediction on the Model page.")
    else:
        df_u = pd.DataFrame(usage)
        df_u["risk_prob"] = pd.to_numeric(df_u["risk_prob"], errors="coerce")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Predictions", len(df_u))
        k2.metric("Avg Risk Score", f"{df_u['risk_prob'].mean():.2%}")
        k3.metric("Approve Rate", f"{(df_u['decision']=='Approve').mean():.1%}")
        k4.metric("Reject Rate",  f"{(df_u['decision']=='Reject').mean():.1%}")

        st.dataframe(df_u.sort_values("timestamp", ascending=False), use_container_width=True)

        fig = px.histogram(df_u, x="risk_prob", nbins=20, color="decision",
                           title="Risk Score Distribution",
                           color_discrete_map={"Approve":"green","Manual Review":"orange","Reject":"red"})
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download", df_u.to_csv(index=False), "predictions.csv", "text/csv")

# ── TAB 2: Feedback ───────────────────────────────────────────────────────────
with tabs[1]:
    feedback = get_feedback(limit=500)
    if not feedback:
        st.info("No feedback submitted yet.")
    else:
        df_f = pd.DataFrame(feedback)
        st.dataframe(df_f.sort_values("timestamp", ascending=False), use_container_width=True)

        if len(df_f) >= 3:
            counts = df_f["feedback"].value_counts().reset_index()
            counts.columns = ["feedback", "count"]
            fig_fb = px.pie(counts, names="feedback", values="count",
                            title="Feedback Distribution",
                            color_discrete_map={"correct":"green","incorrect":"red","unsure":"gray"})
            st.plotly_chart(fig_fb, use_container_width=True)
            correct = (df_f["feedback"] == "correct").sum()
            st.metric("Analyst-Validated Accuracy", f"{correct/len(df_f):.1%}")

        st.download_button("Download", df_f.to_csv(index=False), "feedback.csv", "text/csv")

# ── TAB 3: Audit Log ──────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Audit Trail")
    st.caption("All events logged. Inputs are SHA-256 hashed — no PII stored.")
    audit = get_audit(limit=500)
    if not audit:
        st.info("No audit entries yet.")
    else:
        df_a = pd.DataFrame(audit)
        st.dataframe(df_a.sort_values("timestamp", ascending=False), use_container_width=True)
        st.download_button("Download Audit Log", df_a.to_csv(index=False), "audit_log.csv", "text/csv")

# ── TAB 4: Cost-Benefit Tracker ───────────────────────────────────────────────
with tabs[3]:
    st.subheader("Cost-Benefit Analysis")
    st.caption("Simulate the financial impact of model decisions.")

    cb1, cb2, cb3 = st.columns(3)
    with cb1:
        loan_amt     = st.number_input("Avg Loan Amount (Rs)", value=500000, step=50000)
        default_loss = st.slider("Loss Given Default (%)", 20, 100, 60) / 100
    with cb2:
        interest_rate = st.slider("Annual Interest Rate (%)", 5, 30, 12) / 100
        threshold     = st.slider("Decision Threshold", 0.1, 0.9, 0.5, step=0.05)
    with cb3:
        n_apps       = st.number_input("Applications per Month", value=1000, step=100)
        default_rate = st.slider("True Default Rate (%)", 5, 40, 12) / 100

    if st.button("Calculate"):
        n_defaults = int(n_apps * default_rate)
        n_safe     = n_apps - n_defaults
        recall     = max(0.5, 0.97 - threshold * 0.5)
        fpr        = max(0.01, 0.15 - threshold * 0.2)
        tp = int(n_defaults * recall)
        fp = int(n_safe * fpr)
        fn = n_defaults - tp
        tn = n_safe - fp

        revenue          = tn * loan_amt * interest_rate / 12
        losses           = fn * loan_amt * default_loss
        false_reject_cost = fp * loan_amt * interest_rate / 12 * 0.5
        net              = revenue - losses - false_reject_cost

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Monthly Revenue",    f"Rs {revenue:,.0f}")
        r2.metric("Default Losses",     f"Rs {losses:,.0f}", delta=f"-Rs {losses:,.0f}", delta_color="inverse")
        r3.metric("False Reject Cost",  f"Rs {false_reject_cost:,.0f}")
        r4.metric("Net Profit",         f"Rs {net:,.0f}")

        conf_data = pd.DataFrame({
            "Category": ["True Approve","False Reject","Missed Default","True Reject"],
            "Count": [tn, fp, fn, tp],
            "Financial Impact (Rs)": [tn*loan_amt*interest_rate/12,
                                       -fp*loan_amt*interest_rate/12*0.5,
                                       -fn*loan_amt*default_loss,
                                       tp*loan_amt*default_loss]
        })
        st.dataframe(conf_data.style.format({"Financial Impact (Rs)": "Rs {:,.0f}"}), use_container_width=True)
        st.caption("Ref: Basel II Expected Loss = PD x LGD x EAD | https://www.bis.org/publ/bcbs128.htm")

# ── TAB 5: Performance Monitor ────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Prediction Drift Monitor")
    usage = get_predictions(limit=500)
    if len(usage) < 5:
        st.info("Need at least 5 predictions to show trends.")
    else:
        df_u = pd.DataFrame(usage)
        df_u["risk_prob"] = pd.to_numeric(df_u["risk_prob"], errors="coerce")
        df_u["timestamp"] = pd.to_datetime(df_u["timestamp"])
        df_u = df_u.sort_values("timestamp")

        fig_drift = px.line(df_u, x="timestamp", y="risk_prob",
                            title="Risk Score Over Time",
                            labels={"risk_prob": "Risk Probability", "timestamp": "Time"})
        fig_drift.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Decision boundary")
        st.plotly_chart(fig_drift, use_container_width=True)

        rolling = df_u["risk_prob"].rolling(5, min_periods=1).mean().iloc[-1]
        st.metric("Rolling Avg Risk (last 5)", f"{rolling:.2%}")
        st.caption("Significant drift from training distribution signals need for retraining.")
