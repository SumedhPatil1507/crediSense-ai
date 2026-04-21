import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.feedback import load_usage, load_feedback
from src.validation import load_audit

st.set_page_config(layout="wide")
st.title("📋 Logs, Audit & Cost-Benefit")

tabs = st.tabs(["📊 Usage Logs", "📣 Feedback", "🔐 Audit Log",
                "💰 Cost-Benefit Tracker", "📈 Performance Monitor"])

# ── TAB 1: Usage Logs ─────────────────────────────────────────────────────────
with tabs[0]:
    usage = load_usage()
    if not usage:
        st.info("No predictions logged yet.")
    else:
        df_u = pd.DataFrame(usage)
        df_u["risk_prob"] = df_u["risk_prob"].astype(float)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Predictions", len(df_u))
        k2.metric("Avg Risk Score", f"{df_u['risk_prob'].mean():.2%}")
        k3.metric("Approve Rate", f"{(df_u['decision']=='Approve').mean():.1%}")
        k4.metric("Reject Rate", f"{(df_u['decision']=='Reject').mean():.1%}")

        st.dataframe(df_u.sort_values("timestamp", ascending=False), use_container_width=True)

        fig = px.histogram(df_u, x="risk_prob", nbins=20, color="decision",
                           title="Risk Score Distribution",
                           color_discrete_map={"Approve":"green","Manual Review":"orange","Reject":"red"})
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("⬇ Download", df_u.to_csv(index=False), "usage_log.csv", "text/csv")

# ── TAB 2: Feedback ───────────────────────────────────────────────────────────
with tabs[1]:
    feedback = load_feedback()
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
                            color_discrete_map={"👍 Correct":"green","👎 Incorrect":"red","🤔 Unsure":"gray"})
            st.plotly_chart(fig_fb, use_container_width=True)
            correct = (df_f["feedback"] == "👍 Correct").sum()
            st.metric("Analyst-Validated Accuracy", f"{correct/len(df_f):.1%}")

        st.download_button("⬇ Download", df_f.to_csv(index=False), "feedback_log.csv", "text/csv")

# ── TAB 3: Audit Log ──────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("🔐 Audit Trail")
    st.caption("All prediction and feedback events. Inputs are hashed — no PII stored.")
    audit = load_audit()
    if not audit:
        st.info("No audit entries yet.")
    else:
        df_a = pd.DataFrame(audit)
        st.dataframe(df_a.sort_values("timestamp", ascending=False), use_container_width=True)
        st.caption("input_hash: SHA-256 of raw inputs (non-reversible). user_hash: SHA-256 of user ID.")
        st.download_button("⬇ Download Audit Log", df_a.to_csv(index=False),
                           "audit_log.csv", "text/csv")

# ── TAB 4: Cost-Benefit Tracker ───────────────────────────────────────────────
with tabs[3]:
    st.subheader("💰 Cost-Benefit Analysis")
    st.caption("Simulate the financial impact of the model's decisions.")

    cb1, cb2, cb3 = st.columns(3)
    with cb1:
        loan_amt = st.number_input("Avg Loan Amount (₹)", value=500000, step=50000)
        default_loss = st.slider("Loss Given Default (%)", 20, 100, 60) / 100
    with cb2:
        interest_rate = st.slider("Annual Interest Rate (%)", 5, 30, 12) / 100
        threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5, step=0.05)
    with cb3:
        n_applications = st.number_input("Applications per Month", value=1000, step=100)
        default_rate = st.slider("True Default Rate (%)", 5, 40, 12) / 100

    if st.button("📊 Calculate"):
        n_defaults = int(n_applications * default_rate)
        n_safe = n_applications - n_defaults

        # Simplified model: assume AUC ~0.97 → at threshold 0.5, ~85% recall, ~90% precision
        recall = max(0.5, 0.97 - threshold * 0.5)
        precision = max(0.4, 0.85 + (threshold - 0.3) * 0.3)
        fpr = max(0.01, 0.15 - threshold * 0.2)

        tp = int(n_defaults * recall)
        fp = int(n_safe * fpr)
        fn = n_defaults - tp
        tn = n_safe - fp

        approved = tn + fn
        rejected = tp + fp

        revenue = approved * loan_amt * interest_rate / 12
        losses = fn * loan_amt * default_loss
        false_reject_cost = fp * loan_amt * interest_rate / 12 * 0.5
        net = revenue - losses - false_reject_cost

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Monthly Revenue", f"₹{revenue:,.0f}")
        r2.metric("Default Losses", f"₹{losses:,.0f}", delta=f"-₹{losses:,.0f}", delta_color="inverse")
        r3.metric("False Reject Cost", f"₹{false_reject_cost:,.0f}")
        r4.metric("Net Profit", f"₹{net:,.0f}", delta=f"{'+'if net>0 else ''}₹{net:,.0f}")

        st.markdown("---")
        conf_data = pd.DataFrame({
            "Category": ["True Approve (TP safe)", "False Reject (FP)", "Missed Default (FN)", "True Reject (TP risk)"],
            "Count": [tn, fp, fn, tp],
            "Financial Impact (₹)": [tn*loan_amt*interest_rate/12, -fp*loan_amt*interest_rate/12*0.5,
                                      -fn*loan_amt*default_loss, tp*loan_amt*default_loss]
        })
        st.dataframe(conf_data.style.format({"Financial Impact (₹)": "₹{:,.0f}"}),
                     use_container_width=True)
        st.caption("📚 Based on simplified credit P&L model. "
                   "Ref: [Basel II Expected Loss Framework](https://www.bis.org/publ/bcbs128.htm)")

# ── TAB 5: Performance Monitor ────────────────────────────────────────────────
with tabs[4]:
    st.subheader("📈 Prediction Drift Monitor")
    st.caption("Track how risk scores change over time — early warning for model drift.")
    usage = load_usage()
    if len(usage) < 5:
        st.info("Need at least 5 predictions to show trends. Make more predictions on the Model page.")
    else:
        df_u = pd.DataFrame(usage)
        df_u["risk_prob"] = df_u["risk_prob"].astype(float)
        df_u["timestamp"] = pd.to_datetime(df_u["timestamp"])
        df_u = df_u.sort_values("timestamp")

        fig_drift = px.line(df_u, x="timestamp", y="risk_prob",
                            title="Risk Score Over Time",
                            labels={"risk_prob": "Risk Probability", "timestamp": "Time"})
        fig_drift.add_hline(y=0.5, line_dash="dash", line_color="red",
                            annotation_text="Decision boundary")
        st.plotly_chart(fig_drift, use_container_width=True)

        rolling_mean = df_u["risk_prob"].rolling(5, min_periods=1).mean().iloc[-1]
        st.metric("Rolling Avg Risk (last 5)", f"{rolling_mean:.2%}",
                  help="If this drifts significantly from training distribution, consider retraining.")
        st.caption("📚 [Concept Drift in ML — Gama et al.](https://dl.acm.org/doi/10.1145/2523813)")
