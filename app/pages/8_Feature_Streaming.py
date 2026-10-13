"""Interactive view of Feast feature history from offline, uploaded, or live sources."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import sys

st.set_page_config(page_title="Feature Streaming | CrediSense AI", page_icon="📡", layout="wide")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.dashboard_ui import configure_dashboard
configure_dashboard()

FEATURE_PATH = ROOT / "data" / "credit_feature_events.parquet"
FEATURES = ["rolling_credit_utilization", "payment_velocity", "debt_to_income_ratio"]
REQUIRED = {"applicant_id", "event_timestamp", *FEATURES}


def configured_secret(name: str, default: str = "") -> str:
    value = os.getenv(name, "")
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


def read_event_file(upload) -> pd.DataFrame:
    name = upload.name.lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(upload)
    return pd.read_csv(upload)


def fetch_live_events(url: str, token: str = "") -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {token}", "X-API-Key": token} if token else {}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        payload = payload.get("events", payload.get("features", payload.get("data", [])))
    if not isinstance(payload, list):
        raise ValueError("Endpoint response must be a JSON list or contain an events/features/data list.")
    return pd.DataFrame(payload)


def demo_events() -> pd.DataFrame:
    rows = []
    times = pd.date_range("2025-01-01", periods=8, freq="30D", tz="UTC")
    for person in range(1, 7):
        for index, event_time in enumerate(times):
            rows.append({
                "applicant_id": f"demo-applicant-{person:03d}",
                "pan_hash": f"synthetic-pan-hash-{person:03d}",
                "event_timestamp": event_time,
                "rolling_credit_utilization": min(0.12 * person + 0.035 * index, 0.95),
                "payment_velocity": max(0.99 - 0.055 * person - 0.018 * index, 0.35),
                "debt_to_income_ratio": 0.11 * person + 0.025 * index,
            })
    return pd.DataFrame(rows)


def validate_events(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = REQUIRED.difference(frame.columns)
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(sorted(missing)))
    frame["event_timestamp"] = pd.to_datetime(frame["event_timestamp"], utc=True, errors="coerce")
    if frame["event_timestamp"].isna().any():
        raise ValueError("Some event_timestamp values are not valid timestamps.")
    for feature in FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
        if frame[feature].isna().any():
            raise ValueError(f"Feature '{feature}' contains missing or non-numeric values.")
    frame["applicant_id"] = frame["applicant_id"].astype(str)
    return frame.sort_values("event_timestamp").reset_index(drop=True)


st.markdown("""
<style>
.hero{padding:1.15rem 1.45rem;border-radius:16px;background:linear-gradient(115deg,#101a32,#17496b 62%,#16827e);color:#f7fbff}
.hero h1{margin:0}.hero p{margin:.35rem 0 0;color:#d8e9f2}
</style>
<div class="hero"><h1>📡 Feature Streaming</h1><p>Explore timestamped credit features from the Feast point-in-time feature store.</p></div>
""", unsafe_allow_html=True)

live_url_default = configured_secret("CREDISENSE_LIVE_FEATURES_URL")
sources = ["Feast offline parquet", "Upload CSV or Parquet", "Live feature endpoint", "Synthetic demo"]
if FEATURE_PATH.exists():
    default_source = 0
elif live_url_default:
    default_source = 2
else:
    default_source = 1
source = st.radio("Feature event source", sources, index=default_source, horizontal=True)
frame = pd.DataFrame()
source_label = ""

if source == "Feast offline parquet":
    if FEATURE_PATH.exists():
        try:
            frame = pd.read_parquet(FEATURE_PATH)
            source_label = f"Feast offline source · {FEATURE_PATH.name}"
        except Exception as exc:
            st.error(f"Could not read the Feast parquet file: {exc}")
    else:
        st.info("No local Feast event file is present. Generate one with `python scripts/generate_demo_feature_events.py`, upload an event file, or connect the live endpoint.")
elif source == "Upload CSV or Parquet":
    upload = st.file_uploader("Upload timestamped Feast feature events", type=["csv", "parquet"],
                              help="Required: applicant_id, event_timestamp, rolling_credit_utilization, payment_velocity, debt_to_income_ratio. pan_hash and created_timestamp are optional for visualization.")
    if upload is not None:
        try:
            frame = read_event_file(upload)
            source_label = f"Uploaded event file · {upload.name}"
        except Exception as exc:
            st.error(f"Could not read the uploaded feature file: {exc}")
elif source == "Live feature endpoint":
    live_url = st.text_input("Live event endpoint URL", value=live_url_default, placeholder="https://your-service/features/events")
    live_token = st.text_input("Bearer token (optional)", value=configured_secret("CREDISENSE_LIVE_FEATURES_TOKEN"), type="password")
    if st.button("Fetch latest feature events", type="primary"):
        if not live_url:
            st.warning("Set CREDISENSE_LIVE_FEATURES_URL in Streamlit secrets or enter an endpoint URL.")
        else:
            try:
                st.session_state["live_feature_events"] = fetch_live_events(live_url, live_token)
                st.session_state["live_feature_source"] = live_url
            except Exception as exc:
                st.error(f"Live feature fetch failed: {exc}")
    frame = st.session_state.get("live_feature_events", pd.DataFrame())
    if not frame.empty:
        source_label = "Live feature endpoint · " + st.session_state.get("live_feature_source", "configured source")
elif source == "Synthetic demo":
    frame = demo_events()
    source_label = "Synthetic demonstration data"
    st.warning("This is generated demo history, not applicant or Kafka data.")

if not frame.empty:
    try:
        frame = validate_events(frame)
    except Exception as exc:
        st.error(f"Feature event data is invalid: {exc}")
        st.stop()

    st.success(f"Loaded {len(frame):,} events from {source_label}.")
    applicants = sorted(frame["applicant_id"].unique())
    with st.sidebar:
        st.subheader("Explore events")
        selected_applicants = st.multiselect("Applicants", applicants, default=applicants[: min(3, len(applicants))])
        selected_features = st.multiselect("Feature series", FEATURES, default=FEATURES)
        show_markers = st.toggle("Show event markers", value=True)

    filtered = frame[frame["applicant_id"].isin(selected_applicants)]
    if not selected_features:
        st.warning("Select at least one feature series to draw the history chart.")
        st.stop()
    if filtered.empty:
        st.warning("Select one or more applicants to explore their history.")
        st.stop()

    latest = filtered.sort_values("event_timestamp").groupby("applicant_id", as_index=False).tail(1)
    metric_cols = st.columns(len(selected_features))
    for column, feature in zip(metric_cols, selected_features):
        column.metric(feature.replace("_", " ").title(), f"{filtered[feature].mean():.3f}",
                      delta=f"latest avg {latest[feature].mean():.3f}")

    long = filtered.melt(id_vars=["applicant_id", "event_timestamp"], value_vars=selected_features,
                         var_name="Feature", value_name="Value")
    fig = px.line(long, x="event_timestamp", y="Value", color="applicant_id", line_dash="Feature",
                  markers=show_markers, hover_data={"event_timestamp": True, "Value": ":.4f"},
                  title="Feature values through event time",
                  labels={"event_timestamp": "Event time (UTC)", "applicant_id": "Applicant"}, template="plotly_dark")
    fig.update_layout(legend_title_text="Applicant / feature", hovermode="x unified", height=520)
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        profile = latest.melt(id_vars="applicant_id", value_vars=selected_features, var_name="Feature", value_name="Value")
        profile_fig = px.bar(profile, x="applicant_id", y="Value", color="Feature", barmode="group",
                             template="plotly_dark", title="Most recent value per applicant")
        st.plotly_chart(profile_fig, use_container_width=True)
    with right:
        st.subheader("Event records")
        st.dataframe(filtered.sort_values("event_timestamp", ascending=False), use_container_width=True, hide_index=True)
    st.download_button("Download displayed events", filtered.to_csv(index=False), "credit_feature_events.csv", "text/csv")
else:
    st.info("No feature events are loaded. Upload your exported events or connect a live endpoint. Synthetic data is available from the source selector for a UI preview.")

with st.expander("Connect Kafka feature events to this page"):
    st.markdown("""
    The Kafka consumer writes each valid application to PostgreSQL audit history and to the Feast Redis online store. This page needs a readable event history, so expose those audited feature events through a protected JSON `GET` endpoint and set these Streamlit secrets:

    ```toml
    CREDISENSE_LIVE_FEATURES_URL = "https://your-service/features/events"
    CREDISENSE_LIVE_FEATURES_TOKEN = "your-read-only-token"
    ```

    Return a JSON array (or an `events`, `features`, or `data` array) with `applicant_id`, timezone-aware `event_timestamp`, `rolling_credit_utilization`, `payment_velocity`, and `debt_to_income_ratio`. To explore offline history locally, run `python scripts/generate_demo_feature_events.py` for a demo file or create `data/credit_feature_events.parquet` from real timestamped Feast events. The generated demo file is synthetic.
    """)

with st.expander("How point-in-time retrieval prevents leakage"):
    st.markdown("Each training/application row carries an `event_timestamp`. Feast selects the newest feature event at or before that timestamp for the matching entity keys. Later events cannot flow backward into an earlier snapshot. `created_timestamp` is also used to exclude late-arriving corrections during historical retrieval.")
