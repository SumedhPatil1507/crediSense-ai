import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

import streamlit as st
from app.ui import configure_dashboard
st.set_page_config(page_title="CrediSense AI", layout="wide", page_icon="💳")
configure_dashboard()

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

st.markdown("""
<style>
.hero { padding: 1.4rem 1.7rem; border-radius: 18px; color: #f7f9ff;
        background: linear-gradient(118deg, #101a32 0%, #173e61 55%, #167d83 100%); }
.hero h1 { margin: 0; font-size: 2.2rem; }
.hero p { color: #d4e7f3; margin: .35rem 0 0; font-size: 1.02rem; }
.subtle { color: #8ca4ba; font-size: .88rem; }
</style>
<div class="hero"><h1>CrediSense AI</h1>
<p>Credit risk exploration · explainability · regulatory document review</p></div>
""", unsafe_allow_html=True)
st.caption("Interactive research prototype. Outputs are not validated for real underwriting decisions.")

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
    st.info("**Model lab** — Explore predictions, uncertainty ranges, scenarios, and thresholds")
    st.success("**Feature streaming** — Inspect timestamped features with interactive Plotly charts")
with c2:
    st.warning("**Explainability** — Inspect SHAP drivers and subgroup screening summaries")
    st.error("**Regulatory copilot** — Review retrieved source passages with chunk citations")
with c3:
    st.info("**Monitoring** — Review local prediction logs, feedback, and drift indicators")
    st.success("**Operations** — Explore review queue, model registry, and shadow comparisons")

st.markdown("---")
col_l, col_r = st.columns(2)
with col_l:
    st.markdown("""
    **Explore the app**
    - Interactive Plotly views across model analysis, explanation, operations, and feature history
    - Point-in-time feature-store and Kafka streaming examples
    - Retrieval-backed regulatory Q&A with fail-closed unsupported-claim handling
    - API inference and optional graph-risk attribution
    """)
with col_r:
    st.markdown("""
    **Local API**
    - `POST /api/v1/predict` — Single prediction with CI + adverse action
    - `POST /api/v1/predict/batch` — Batch (up to 500)
    - `POST /api/v1/explain` — SHAP top features
    - `POST /api/v1/feedback` — Log analyst feedback
    - `GET /api/v1/metrics` — Aggregate stats
    - `GET /health` — System health
    - `GET /metrics` — Prometheus metrics (if enabled)
    - Swagger UI: `/docs` | ReDoc: `/redoc`
    """)

st.markdown("---")
st.subheader("Explore the training data")
st.caption("Interactive class counts from the bundled sample dataset; this is not a model performance measure.")
try:
    import pandas as pd
    import plotly.express as px

    @st.cache_data(show_spinner=False)
    def _load_class_counts():
        frame = pd.read_csv(BASE_DIR / "data" / "loan_cleaned.csv", sep="\t", usecols=["Risk_Flag"])
        counts = frame["Risk_Flag"].value_counts().rename_axis("Dataset label").reset_index(name="Applicants")
        counts["Dataset label"] = counts["Dataset label"].map({0: "Non-default label", 1: "Default label"}).fillna(counts["Dataset label"].astype(str))
        return counts

    counts = _load_class_counts()
    chart_col, note_col = st.columns([1.25, 1])
    with chart_col:
        class_fig = px.pie(counts, names="Dataset label", values="Applicants", hole=0.62,
                           color_discrete_sequence=["#45c2b1", "#ff8b72"],
                           template="plotly_dark", title="Sample label mix")
        class_fig.update_traces(textposition="inside", textinfo="label+percent",
                                hovertemplate="%{label}<br>Applicants: %{value:,}<br>Share: %{percent}<extra></extra>")
        class_fig.update_layout(height=350, legend_title_text="Dataset labels")
        st.plotly_chart(class_fig, use_container_width=True)
    with note_col:
        st.metric("Sample rows", f"{int(counts['Applicants'].sum()):,}")
        st.markdown("The chart reports class counts in the bundled training dataset. It does not show model accuracy, approval quality, calibration, or fairness. Use the Model and Explainability pages to inspect their respective exploratory views.")
except (FileNotFoundError, ValueError, ImportError) as exc:
    st.info(f"Training data chart is unavailable in this deployment: {exc}")

st.sidebar.success("Select a page above")
st.sidebar.markdown("---")
st.sidebar.caption("CrediSense AI v1.0 | LightGBM + FastAPI + SQLite")
