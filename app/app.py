import streamlit as st
from src.model_registry import get_active_version, get_current_model_hash
from src.database import db_status
from src.hitl_queue import queue_stats
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

st.set_page_config(page_title="CrediSense AI", layout="wide", page_icon="💳")

st.title("CrediSense AI")
st.markdown("### Production-Grade Credit Risk Scoring System")
st.markdown("---")

# System status bar
db = db_status()
active = get_active_version()
model_hash = get_current_model_hash()
q_stats = queue_stats()

s1, s2, s3, s4 = st.columns(4)
s1.metric("DB Backend", "Supabase" if db["backend"] == "supabase" else "CSV Fallback",
          delta="persistent" if db["backend"] == "supabase" else "ephemeral")
s2.metric("Model Version", active["version_id"] if active else "Unregistered",
          delta=f"hash: {model_hash}")
s3.metric("HITL Queue", q_stats["pending"],
          delta=f"{q_stats['pending']} pending" if q_stats["pending"] > 0 else "clear",
          delta_color="inverse" if q_stats["pending"] > 0 else "normal")
s4.metric("Total Resolved", q_stats["resolved"])

st.markdown("---")

c1, c2, c3 = st.columns(3)
with c1:
    st.info("**EDA** — Dataset analysis, live macro indicators, RSS news feed")
    st.success("**Model** — Predict with CI, what-if simulator, model comparison, threshold analysis")
with c2:
    st.warning("**Explainability** — SHAP summary, waterfall, dependence, LIME")
    st.error("**Chatbot** — Risk assistant with real inputs, Q&A knowledge base")
with c3:
    st.info("**Logs** — Usage logs, feedback, audit trail, cost-benefit, drift monitor")
    st.success("**Operations** — HITL queue, PSI/CSI drift, shadow mode, model registry, alerts")

st.markdown("---")
col_l, col_r = st.columns(2)
with col_l:
    st.markdown("""
    **System Capabilities:**
    - LightGBM credit risk model (ROC-AUC ~0.97, Gini ~0.94, KS ~0.75)
    - Bootstrap 95% confidence intervals on every prediction
    - ECOA/FCRA adverse action notices for rejections
    - SHAP + LIME explainability (global + local)
    - Human-in-the-loop queue for borderline cases
    - PSI/CSI drift monitoring with auto-alerts
    - Shadow mode for challenger model comparison
    - Model version registry with performance tracking
    """)
with col_r:
    st.markdown("""
    **API Endpoints (FastAPI):**
    - `POST /api/v1/predict` — Single prediction
    - `POST /api/v1/predict/batch` — Batch (up to 500)
    - `POST /api/v1/explain` — SHAP explanation
    - `POST /api/v1/feedback` — Log analyst feedback
    - `GET /api/v1/metrics` — Aggregate metrics
    - `GET /health` — System health check
    - Swagger UI: `/docs` | ReDoc: `/redoc`
    """)

st.sidebar.success("Select a page above")
st.sidebar.markdown("---")
st.sidebar.caption("CrediSense AI v1.0 | LightGBM + FastAPI + Supabase")
