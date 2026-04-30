"""
Operations Dashboard — Production monitoring for analysts and risk managers.
Covers: HITL Queue, Model Drift (PSI/CSI), Shadow Mode, Model Registry, Alerts.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.hitl_queue   import get_queue, resolve, queue_stats, enqueue
from src.drift_monitor import compute_psi, interpret_psi, compute_csi
from src.shadow_mode  import get_shadow_log, shadow_agreement_rate
from src.model_registry import get_all_versions, get_active_version, get_current_model_hash
from src.database     import get_predictions
from src.alerts       import alert, WEBHOOK_URL, ALERT_EMAIL

st.set_page_config(layout="wide")
st.title("Operations Center")
st.caption("HITL Queue | Drift Monitor | Shadow Mode | Model Registry | Alerts")

tabs = st.tabs([
    "Human Review Queue",
    "Drift Monitor (PSI/CSI)",
    "Shadow Mode",
    "Model Registry",
    "Alerts & Webhooks",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — HITL QUEUE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    stats = queue_stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Pending Review", stats["pending"],
              delta=f"+{stats['pending']}" if stats["pending"] > 0 else None,
              delta_color="inverse")
    c2.metric("Resolved",  stats["resolved"])
    c3.metric("Total",     stats["total"])

    if stats["pending"] > 0:
        st.warning(f"{stats['pending']} cases awaiting analyst review.")

    st.markdown("---")
    view = st.radio("View", ["Pending", "Resolved", "All"], horizontal=True)
    status_map = {"Pending": "pending", "Resolved": "resolved", "All": "all"}
    queue = get_queue(status=status_map[view])

    if not queue:
        st.info(f"No {view.lower()} cases.")
    else:
        df_q = pd.DataFrame(queue)
        df_q["risk_prob"] = pd.to_numeric(df_q["risk_prob"], errors="coerce")

        # Resolve form
        if view == "Pending" and len(df_q) > 0:
            st.subheader("Resolve a Case")
            item_ids = df_q["id"].tolist()
            sel_id = st.selectbox("Select case ID", item_ids)
            sel_row = df_q[df_q["id"] == sel_id].iloc[0]

            from src.hitl_queue import resolve as resolve_item

            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown(f"**Risk Probability:** {float(sel_row['risk_prob']):.2%}")
                st.markdown(f"**Reason:** {sel_row['reason']}")
                st.markdown(f"**CI:** [{sel_row['ci_lower']}, {sel_row['ci_upper']}]")
                # Rule-based recommendation
                rp = float(sel_row['risk_prob'])
                rec = "Approve" if rp < 0.35 else "Reject" if rp > 0.55 else "Request More Info"
                st.info(f"Suggested Decision: **{rec}** (based on risk score {rp:.1%})")
            with rc2:
                analyst_dec = st.selectbox("Analyst Decision",
                                            ["Approve", "Reject", "Escalate", "Request More Info"],
                                            key="hitl_decision")
                analyst_notes = st.text_area("Notes", placeholder="Reasoning for decision...",
                                              key="hitl_notes")

            if st.button("Submit Decision", type="primary", key="hitl_submit"):
                resolve(sel_id, analyst_dec, analyst_notes)
                st.success(f"Case {sel_id} resolved as: {analyst_dec}")
                st.rerun()

        st.markdown("---")
        st.dataframe(df_q, use_container_width=True)

        if len(df_q) > 0 and "risk_prob" in df_q.columns:
            fig = px.histogram(df_q, x="risk_prob", nbins=15,
                               title="Risk Score Distribution in Queue",
                               color_discrete_sequence=["#ffc107"])
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DRIFT MONITOR
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Population Stability Index (PSI)")
    st.caption("PSI < 0.1: Stable | 0.1-0.2: Warning | > 0.2: Retrain needed")

    preds = get_predictions(limit=500)
    if len(preds) < 20:
        st.info("Need at least 20 live predictions to compute PSI. Make more predictions on the Model page.")
    else:
        df_live = pd.DataFrame(preds)
        df_live["risk_prob"] = pd.to_numeric(df_live["risk_prob"], errors="coerce").dropna()
        live_scores = df_live["risk_prob"].dropna().values

        with st.spinner("Loading training distribution..."):
            from src.drift_monitor import get_training_score_distribution
            train_scores = get_training_score_distribution()

        if len(train_scores) == 0:
            st.warning("Could not load training distribution.")
        else:
            psi = compute_psi(train_scores, live_scores)
            status, msg = interpret_psi(psi)

            p1, p2, p3 = st.columns(3)
            p1.metric("PSI Score", f"{psi:.4f}")
            p2.metric("Status", status.upper())
            p3.metric("Training Samples", f"{len(train_scores):,}")

            if status == "stable":   st.success(f"Model is stable. {msg}")
            elif status == "warning": st.warning(f"Drift warning. {msg}")
            else:                     st.error(f"Critical drift. {msg}")

            # Distribution comparison
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=train_scores, name="Training",
                                             opacity=0.6, nbinsx=30,
                                             marker_color="royalblue",
                                             histnorm="probability"))
            fig_dist.add_trace(go.Histogram(x=live_scores, name="Live",
                                             opacity=0.6, nbinsx=30,
                                             marker_color="darkorange",
                                             histnorm="probability"))
            fig_dist.update_layout(barmode="overlay",
                                   title="Training vs Live Score Distribution",
                                   xaxis_title="Risk Probability",
                                   yaxis_title="Density", height=380)
            st.plotly_chart(fig_dist, use_container_width=True)

            # Auto-alert
            from src.alerts import check_and_alert_drift
            alert_result = check_and_alert_drift(psi)
            if alert_result and alert_result.get("any_sent"):
                st.info("Alert sent to configured channels.")

            st.markdown("---")
            st.subheader("Self-Healing (Automated Retraining)")
            if status == "critical":
                st.error("Critical drift detected. The system recommends retraining.")
                if st.button("Trigger Retraining", type="primary", key="retrain_critical"):
                    st.warning("Retraining requires running `python -m src.train` locally or via CI pipeline. "
                               "Streamlit Cloud does not support long-running training jobs.")
            else:
                st.info("Model is within acceptable drift thresholds. Retraining not required.")
                if st.button("Force Retraining", key="retrain_force"):
                    st.warning("Run `python -m src.train` locally to retrain and push the new model.pkl.")

    st.markdown("---")
    st.subheader("Characteristic Stability Index (CSI) — Feature-Level Drift")
    st.caption("Which individual features are drifting most.")

    if len(preds) >= 20:
        df_live_feats = pd.DataFrame(preds)
        numeric_feats = ["income_norm", "age_norm", "experience_norm"]
        available = [f for f in numeric_feats if f in df_live_feats.columns]

        if available:
            from src.data_loader import load_data
            from src.preprocessing import clean_data
            from src.feature_engineering import create_features
            from sklearn.model_selection import train_test_split

            @st.cache_data
            def get_train_feats():
                df = load_data(str(BASE_DIR / "data" / "loan_cleaned.csv"))
                df = clean_data(df)
                df = create_features(df)
                _, X_test, _, _ = train_test_split(
                    df.drop(columns=["Risk_Flag"]), df["Risk_Flag"],
                    test_size=0.2, stratify=df["Risk_Flag"], random_state=42)
                return X_test

            train_df = get_train_feats()
            rename_map = {"income_norm": "Income", "age_norm": "Age", "experience_norm": "Experience"}
            live_renamed = df_live_feats[available].rename(columns=rename_map)
            train_renamed = train_df[[rename_map[f] for f in available if rename_map[f] in train_df.columns]]

            csi_df = compute_csi(train_renamed, live_renamed,
                                  [rename_map[f] for f in available if rename_map[f] in train_renamed.columns])
            if not csi_df.empty:
                fig_csi = px.bar(csi_df, x="psi", y="feature", orientation="h",
                                  color="status",
                                  color_discrete_map={"stable":"green","warning":"orange","critical":"red"},
                                  title="CSI by Feature", text="psi")
                fig_csi.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                st.plotly_chart(fig_csi, use_container_width=True)
                st.dataframe(csi_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SHADOW MODE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Shadow Mode — Champion vs Challenger")
    st.caption("Run a challenger model silently alongside the champion. Compare decisions without affecting live output.")

    shadow_log = get_shadow_log()
    agreement  = shadow_agreement_rate()

    if not shadow_log:
        st.info("No shadow mode data yet.")
        st.markdown("""
        **How to enable Shadow Mode:**
        1. Train a challenger model and save it to `models/challenger.pkl`
        2. Shadow mode will automatically run both models on each prediction
        3. Results are logged here for comparison
        """)
    else:
        df_s = pd.DataFrame(shadow_log)
        df_s["champion_prob"]   = pd.to_numeric(df_s["champion_prob"],   errors="coerce")
        df_s["challenger_prob"] = pd.to_numeric(df_s["challenger_prob"], errors="coerce")

        s1, s2, s3 = st.columns(3)
        s1.metric("Total Shadow Runs", len(df_s))
        s2.metric("Agreement Rate", f"{agreement:.1%}" if agreement else "N/A")
        s3.metric("Disagreements", len(df_s[df_s["agreement"] == "False"]))

        fig_shadow = go.Figure()
        fig_shadow.add_trace(go.Scatter(
            x=df_s["champion_prob"], y=df_s["challenger_prob"],
            mode="markers", marker=dict(
                color=df_s["agreement"].map({"True": "green", "False": "red"}),
                size=6, opacity=0.6
            ), name="Predictions"
        ))
        fig_shadow.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Perfect Agreement",
                                         line=dict(dash="dash", color="gray")))
        fig_shadow.update_layout(title="Champion vs Challenger Probabilities",
                                  xaxis_title="Champion", yaxis_title="Challenger", height=400)
        st.plotly_chart(fig_shadow, use_container_width=True)

        disagreements = df_s[df_s["agreement"] == "False"]
        if len(disagreements) > 0:
            st.subheader("Disagreements")
            st.dataframe(disagreements, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("Model Version Registry")

    active = get_active_version()
    current_hash = get_current_model_hash()

    if active:
        st.success(f"Active version: **{active['version_id']}** | Hash: `{active['model_hash']}`")
        if active["model_hash"] != current_hash:
            st.warning(f"Deployed model hash ({current_hash}) differs from registry. Model may have been updated without registration.")
    else:
        st.warning("No registered model versions found.")
        st.caption(f"Current model hash: `{current_hash}`")

    versions = get_all_versions()
    if versions:
        df_v = pd.DataFrame(versions)
        cols_show = [c for c in ["version_id","timestamp","status","model_hash","description"] if c in df_v.columns]
        st.dataframe(df_v[cols_show], use_container_width=True)

        if "metrics" in df_v.columns:
            st.subheader("Performance Across Versions")
            metrics_rows = []
            for _, row in df_v.iterrows():
                if isinstance(row.get("metrics"), dict):
                    m = {"version": row["version_id"], **row["metrics"]}
                    metrics_rows.append(m)
            if metrics_rows:
                df_m = pd.DataFrame(metrics_rows)
                if "AUC" in df_m.columns:
                    fig_v = px.line(df_m, x="version", y="AUC", markers=True,
                                    title="ROC-AUC Across Model Versions")
                    st.plotly_chart(fig_v, use_container_width=True)
    else:
        st.info("No versions registered yet.")
        st.markdown("""
        **Register the current model:**
        ```python
        from src.model_registry import register_model
        register_model("models/model.pkl",
                       metrics={"AUC": 0.97, "Gini": 0.94, "KS": 0.75},
                       description="LightGBM v1 — initial production model")
        ```
        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ALERTS & WEBHOOKS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("Alert Configuration")

    a1, a2 = st.columns(2)
    with a1:
        st.markdown("**Webhook Status**")
        if WEBHOOK_URL:
            st.success("Webhook configured")
        else:
            st.warning("No webhook configured. Set `ALERT_WEBHOOK_URL` in environment.")
            st.caption("Supports Slack, Teams, Discord, or any HTTP POST endpoint.")

    with a2:
        st.markdown("**Email Status**")
        if ALERT_EMAIL:
            st.success(f"Email alerts → {ALERT_EMAIL}")
        else:
            st.warning("No email configured. Set `ALERT_EMAIL`, `SMTP_USER`, `SMTP_PASS`.")

    st.markdown("---")
    st.subheader("Send Test Alert")
    test_msg = st.text_input("Message", value="Test alert from CrediSense AI")
    test_level = st.selectbox("Level", ["info", "warning", "critical"])
    if st.button("Send Test Alert"):
        result = alert("Test Alert", test_msg, test_level)
        if result["any_sent"]:
            st.success(f"Alert sent — webhook: {result['webhook']}, email: {result['email']}")
        else:
            st.warning("No channels configured. Alert not sent.")
            st.caption("Configure ALERT_WEBHOOK_URL or SMTP settings in your environment.")

    st.markdown("---")
    st.subheader("Alert Rules")
    st.markdown("""
    | Trigger | Threshold | Level |
    |---------|-----------|-------|
    | PSI drift | > 0.2 | Critical |
    | PSI drift | 0.1 - 0.2 | Warning |
    | HITL queue backlog | > 20 pending | Warning |
    | Model hash mismatch | Any | Info |
    """)
    st.caption("Alerts fire automatically when predictions are made and thresholds are exceeded.")
