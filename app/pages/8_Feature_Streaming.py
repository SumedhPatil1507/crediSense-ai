"""Interactive view of point-in-time credit feature history."""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Feature Streaming | CrediSense AI", page_icon="📡", layout="wide")

ROOT = Path(__file__).resolve().parents[2]
FEATURE_PATH = ROOT / "data" / "credit_feature_events.parquet"
FEATURES = [
    "rolling_credit_utilization",
    "payment_velocity",
    "debt_to_income_ratio",
]


@st.cache_data(ttl=30)
def load_feature_events() -> tuple[pd.DataFrame, bool]:
    if FEATURE_PATH.exists():
        frame = pd.read_parquet(FEATURE_PATH)
        return frame, False

    # Keep the page explorable before the first Kafka events or demo seed exist.
    events = []
    times = pd.date_range("2025-01-01", periods=8, freq="30D", tz="UTC")
    for person in range(1, 7):
        for index, event_time in enumerate(times):
            events.append({
                "applicant_id": f"demo-applicant-{person:03d}",
                "pan_hash": f"synthetic-pan-hash-{person:03d}",
                "event_timestamp": event_time,
                "rolling_credit_utilization": min(0.12 * person + 0.035 * index, 0.95),
                "payment_velocity": max(0.99 - 0.055 * person - 0.018 * index, 0.35),
                "debt_to_income_ratio": 0.11 * person + 0.025 * index,
            })
    return pd.DataFrame(events), True


st.title("📡 Feature Streaming")
st.caption("Explore timestamped credit features used by the Feast point-in-time feature store.")

frame, is_demo = load_feature_events()
if is_demo:
    st.info("Showing synthetic demo history. Generate the Feast parquet file or ingest Kafka events to view your feature data.")
else:
    st.success(f"Loaded {len(frame):,} timestamped feature events from the local Feast offline source.")

required = {"applicant_id", "event_timestamp", *FEATURES}
missing = required.difference(frame.columns)
if missing:
    st.error("Feature event file is missing required columns: " + ", ".join(sorted(missing)))
    st.stop()

frame["event_timestamp"] = pd.to_datetime(frame["event_timestamp"], utc=True)
frame = frame.sort_values("event_timestamp")
applicants = sorted(frame["applicant_id"].astype(str).unique())
with st.sidebar:
    st.subheader("Explore events")
    selected_applicants = st.multiselect("Applicants", applicants, default=applicants[: min(3, len(applicants))])
    selected_features = st.multiselect("Feature series", FEATURES, default=FEATURES)
    show_markers = st.toggle("Show event markers", value=True)

filtered = frame[frame["applicant_id"].astype(str).isin(selected_applicants)]
if not selected_features:
    st.warning("Select at least one feature series to draw the history chart.")
    st.stop()
if filtered.empty:
    st.warning("Select one or more applicants to explore their history.")
    st.stop()

latest = filtered.sort_values("event_timestamp").groupby("applicant_id", as_index=False).tail(1)
metric_cols = st.columns(3)
for column, feature, label in zip(metric_cols, selected_features, selected_features):
    column.metric(label.replace("_", " ").title(), f"{filtered[feature].mean():.3f}",
                  delta=f"latest avg {latest[feature].mean():.3f}")

long = filtered.melt(
    id_vars=["applicant_id", "event_timestamp"],
    value_vars=selected_features,
    var_name="Feature",
    value_name="Value",
)
fig = px.line(
    long, x="event_timestamp", y="Value", color="applicant_id", line_dash="Feature",
    markers=show_markers, hover_data={"event_timestamp": True, "Value": ":.4f"},
    title="Feature values through event time",
    labels={"event_timestamp": "Event time (UTC)", "applicant_id": "Applicant"},
    template="plotly_dark",
)
fig.update_layout(legend_title_text="Applicant / feature", hovermode="x unified", height=520)
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns([1, 1])
with left:
    st.subheader("Latest feature profile")
    profile = latest.melt(id_vars="applicant_id", value_vars=selected_features,
                          var_name="Feature", value_name="Value")
    profile_fig = px.bar(profile, x="applicant_id", y="Value", color="Feature",
                         barmode="group", template="plotly_dark", title="Most recent value per applicant")
    st.plotly_chart(profile_fig, use_container_width=True)
with right:
    st.subheader("Event records")
    st.dataframe(filtered.sort_values("event_timestamp", ascending=False),
                 use_container_width=True, hide_index=True)

with st.expander("How point-in-time retrieval prevents leakage"):
    st.markdown(
        "Each training/application row carries an `event_timestamp`. Feast selects the newest "
        "feature event at or before that timestamp for the matching entity keys. A later "
        "correction or payment event therefore cannot flow backward into an earlier snapshot. "
        "Use `tests/test_feature_store.py` to exercise this behavior against the local offline store."
    )
