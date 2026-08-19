import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

import streamlit as st

# ── Cached resource loaders — loaded once, shared across all pages ─────────────

@st.cache_resource(show_spinner=False)
def _load_model():
    """Load LightGBM pipeline once. cache_resource keeps it in memory across reruns."""
    import joblib
    return joblib.load(BASE_DIR / "models" / "model.pkl")


@st.cache_resource(show_spinner=False)
def _load_columns():
    """Load feature columns once."""
    import json
    with open(BASE_DIR / "models" / "columns.json") as f:
        return json.load(f)


@st.cache_data(ttl=300, show_spinner=False)
def _load_system_status():
    """Cache system status for 5 minutes — these don't change often."""
    from src.model_registry import get_active_version, get_current_model_hash
    from src.database import db_status
    from src.hitl_queue import queue_stats
    return {
        "db": db_status(),
        "active": get_active_version(),
        "model_hash": get_current_model_hash(),
        "queue": queue_stats(),
    }


# Pre-warm the model on app startup (avoids cold-start lag on first prediction)
_load_model()
_load_columns()

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="CrediSense AI", layout="wide", page_icon="💳")
st.title("CrediSense AI")
st.markdown("### Production-Grade Credit Risk Scoring System")
st.markdown("---")

# ── System status bar ──────────────────────────────────────────────────────────

status = _load_system_status()
db     = status["db"]
active = status["active"]
model_hash = status["model_hash"]
q_stats    = status["queue"]

s1, s2, s3, s4 = st.columns(4)
s1.metric("DB Backend", "SQLite", delta="persistent local DB")
s2.metric("Model Version",
          active["version_id"] if active else "Unregistered",
          delta=f"hash: {model_hash}")
s3.metric("HITL Queue", q_stats["pending"],
          delta=f"{q_stats['pending']} pending" if q_stats["pending"] > 0 else "clear",
          delta_color="inverse" if q_stats["pending"] > 0 else "normal")
s4.metric("Total Resolved", q_stats["resolved"])

st.markdown("---")

# ── Page cards ─────────────────────────────────────────────────────────────────

c1, c2, c3 = st.columns(3)
with c1:
    st.info("**EDA** — Dataset analysis, live macro indicators, RSS news feed")
    st.success("**Model** — Predict with CI, what-if simulator, model comparison, threshold analysis")
with c2:
    st.warning("**Explainability** — SHAP beeswarm, waterfall, interaction heatmap, fairness audit")
    st.error("**Chatbot** — Risk assistant with real inputs (LPA/age/years), Q&A knowledge base")
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
    - AES-256 field-level encryption + strict PII masking
    - ECOA/FCRA adverse action notices for rejections
    - SHAP explainability + fairness audit (Disparate Impact Ratio)
    - Human-in-the-loop queue for borderline cases
    - PSI/CSI drift monitoring with auto-alerts
    - RBI IT Framework + DPDP Act 2023 compliance module
    """)
with col_r:
    st.markdown("""
    **API Endpoints (FastAPI):**
    - `POST /api/v1/predict` — Single prediction with CI + adverse action
    - `POST /api/v1/predict/batch` — Batch (up to 500)
    - `POST /api/v1/explain` — SHAP top features
    - `POST /api/v1/feedback` — Log analyst feedback
    - `GET /api/v1/metrics` — Aggregate stats
    - `GET /health` — System health
    - `GET /metrics` — Prometheus metrics (if enabled)
    - Swagger UI: `/docs` | ReDoc: `/redoc`
    """)

st.sidebar.success("Select a page above")
st.sidebar.markdown("---")
st.sidebar.caption("CrediSense AI v1.0 | LightGBM + FastAPI + SQLite")
