"""Interactive, source-aware batch scoring and explanation dashboard."""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import THRESHOLD_APPROVE, THRESHOLD_REVIEW
from src.database import get_predictions
from src.explainability import get_explainer_and_values
from src.feature_engineering import create_features
from src.fairness import disparate_impact_ratio, subgroup_metrics
from src.telemetry import traced_predict_proba

st.set_page_config(page_title="Explainability | CrediSense AI", page_icon="📊", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "models" / "model.pkl")


def configured_secret(name: str, default: str = "") -> str:
    value = os.getenv(name, "")
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


def recent_prediction_rows() -> pd.DataFrame:
    """Read the latest records actually scored by this app instance."""
    records = get_predictions(limit=500)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    aliases = {"income_norm": "Income", "age_norm": "Age", "experience_norm": "Experience"}
    df = df.rename(columns=aliases)
    return df


def fetch_live_rows(url: str, token: str = "") -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = requests.get(url, headers=headers, timeout=12)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        payload = payload.get("applications", payload.get("data", []))
    if not isinstance(payload, list):
        raise ValueError("Live endpoint must return a JSON list or an applications/data list.")
    return pd.DataFrame(payload)


def prepare_features(rows: pd.DataFrame, expected_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Align uploaded/live application rows with the trained model's input contract."""
    raw = rows.copy().reset_index(drop=True)
    raw.columns = [str(c).strip() for c in raw.columns]
    required = ["Income", "Age", "Experience"]
    missing_required = [c for c in required if c not in raw.columns]
    if missing_required:
        raise ValueError("Missing required model inputs: " + ", ".join(missing_required))

    defaults = {
        "Married/Single": "single", "House_Ownership": "rented", "Car_Ownership": "no",
        "Profession": "Other", "CITY": "Unknown", "STATE": "Unknown",
        "CURRENT_JOB_YRS": 0.0, "CURRENT_HOUSE_YRS": 0.0,
    }
    for col, value in defaults.items():
        if col not in raw.columns:
            raw[col] = value
    raw["CURRENT_JOB_YRS"] = pd.to_numeric(raw["CURRENT_JOB_YRS"], errors="coerce").fillna(0)
    raw["CURRENT_HOUSE_YRS"] = pd.to_numeric(raw["CURRENT_HOUSE_YRS"], errors="coerce").fillna(0)
    for col in required:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    bad = raw[required].isna().any(axis=1)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} row(s) have missing or non-numeric Income, Age, or Experience values.")

    if "age_group" not in raw.columns:
        raw["age_group"] = pd.cut(raw["Age"], [-np.inf, 0.3, 0.7, np.inf], labels=["Young", "Middle", "Senior"]).astype(str)
    engineered = create_features(raw.copy())
    missing_optional = [c for c in expected_cols if c not in engineered.columns]
    for col in missing_optional:
        engineered[col] = 0
    X = engineered.reindex(columns=expected_cols).copy()
    if len(X) > 500:
        raise ValueError("Please provide at most 500 rows per scoring run.")
    return X, raw, missing_optional


def decision_for(prob: float) -> str:
    if prob < THRESHOLD_APPROVE:
        return "Lower risk — eligible for policy review"
    if prob < THRESHOLD_REVIEW:
        return "Borderline — manual review"
    return "Elevated risk — manual review required"


def explanation_for(prob: float, shap_row: np.ndarray, names: list[str]) -> tuple[str, str, str]:
    order = np.argsort(np.abs(shap_row))[::-1]
    drivers = []
    for i in order:
        if abs(float(shap_row[i])) < 1e-8:
            continue
        name = names[i].replace("num__", "").replace("cat__", "")
        direction = "increased" if shap_row[i] > 0 else "reduced"
        drivers.append(f"{name} {direction} the model score")
        if len(drivers) == 3:
            break
    why = "; ".join(drivers) if drivers else "No material feature contribution was identified."
    recommendation = decision_for(prob)
    caveat = "SHAP describes model contributions, not causes or a complete creditworthiness assessment. Verify source data and apply approved policy before acting."
    return why, recommendation, caveat


st.markdown("""
<style>
.hero {padding:1.2rem 1.5rem;border-radius:16px;background:linear-gradient(115deg,#101a32,#17496b 62%,#16827e);color:#f7fbff}
.hero h1{margin:0}.hero p{margin:.35rem 0 0;color:#d8e9f2}
</style>
<div class="hero"><h1>Explainability & Fairness</h1><p>Score current applicant records, inspect interactive SHAP drivers, and review every recommendation.</p></div>
""", unsafe_allow_html=True)
st.caption("Decision support prototype. Model recommendations are not lending decisions and require human review.")

model = load_model()
expected_cols = list(model.named_steps["preprocessor"].feature_names_in_)
st.subheader("Choose applicant data")
source = st.radio("Data source", ["Upload CSV", "Recent app predictions", "Live application endpoint"], horizontal=True)
source_rows = pd.DataFrame()
if source == "Upload CSV":
    upload = st.file_uploader("Upload current applicant records", type=["csv"], help="Required columns: Income, Age, Experience. Values should use the trained model's normalized 0–1 scale. Optional demographic and stability fields improve attribution fidelity.")
    template = pd.DataFrame([{"Income": 0.42, "Age": 0.31, "Experience": 0.18,
                              "Married/Single": "single", "House_Ownership": "rented",
                              "Car_Ownership": "no", "Profession": "Analyst", "CITY": "Mumbai",
                              "STATE": "Maharashtra", "CURRENT_JOB_YRS": 0.2,
                              "CURRENT_HOUSE_YRS": 0.3}])
    st.download_button("Download CSV template", template.to_csv(index=False), "applicant_upload_template.csv", "text/csv")
    if upload is not None:
        try:
            source_rows = pd.read_csv(upload)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
elif source == "Recent app predictions":
    if st.button("Refresh recent predictions", type="primary"):
        st.rerun()
    source_rows = recent_prediction_rows()
    if source_rows.empty:
        st.info("No live prediction records are available in this app instance yet. Make a prediction on the Model page or upload a CSV.")
    else:
        st.caption("Source: recent rows from this Streamlit instance's prediction log. Cloud instance storage may reset on restart.")
else:
    live_url = st.text_input("Live endpoint URL", value=configured_secret("CREDISENSE_LIVE_APPLICATIONS_URL"), placeholder="https://your-service/applications")
    live_token = st.text_input("Bearer token (optional)", type="password", value=configured_secret("CREDISENSE_LIVE_APPLICATIONS_TOKEN"))
    if st.button("Fetch live applications", type="primary"):
        if not live_url:
            st.warning("Set a URL or configure CREDISENSE_LIVE_APPLICATIONS_URL.")
        else:
            try:
                st.session_state["live_applicant_rows"] = fetch_live_rows(live_url, live_token)
                st.success(f"Fetched {len(st.session_state['live_applicant_rows'])} current application row(s).")
            except Exception as exc:
                st.error(f"Live fetch failed: {exc}")
    source_rows = st.session_state.get("live_applicant_rows", pd.DataFrame())

if not source_rows.empty:
    st.markdown("#### Input preview")
    st.dataframe(source_rows.head(8), use_container_width=True, hide_index=True)
    if st.button("Score and explain all rows", type="primary", use_container_width=True):
        try:
            X, raw_rows, defaulted = prepare_features(source_rows, expected_cols)
            with st.spinner("Scoring records and computing per-applicant SHAP explanations…"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    probs = traced_predict_proba(model, X, operation="explainability_batch")[:, 1]
                _, shap_values, dense_values, feature_names = get_explainer_and_values(model, X)
            results = []
            for i, prob in enumerate(probs):
                why, recommendation, caveat = explanation_for(float(prob), shap_values[i], feature_names)
                applicant_id = raw_rows.iloc[i].get("application_id", raw_rows.iloc[i].get("Id", i + 1))
                results.append({"Applicant": str(applicant_id), "Default probability": float(prob),
                                "Recommendation": recommendation, "Reasoning (top SHAP drivers)": why,
                                "Review note": caveat})
            result_df = pd.DataFrame(results)
            st.session_state["explainability_results"] = result_df
            st.session_state["explainability_X"] = X
            st.session_state["explainability_shap"] = shap_values
            st.session_state["explainability_dense"] = dense_values
            st.session_state["explainability_names"] = feature_names
            st.session_state["explainability_source_rows"] = raw_rows
            st.session_state["explainability_defaulted"] = defaulted
        except Exception as exc:
            st.error(f"Scoring could not be completed: {exc}")

if "explainability_results" in st.session_state:
    result_df = st.session_state["explainability_results"]
    X = st.session_state["explainability_X"]
    sv = st.session_state["explainability_shap"]
    dense_values = st.session_state["explainability_dense"]
    names = st.session_state["explainability_names"]
    raw = st.session_state["explainability_source_rows"]
    defaulted = st.session_state["explainability_defaulted"]
    st.markdown("---")
    st.subheader("Applicant predictions and reasoning")
    if defaulted:
        st.warning("Some optional inputs were absent and defaulted for scoring: " + ", ".join(defaulted) + ". The corresponding explanations may be less informative.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Rows scored", len(result_df))
    m2.metric("Average model probability", f"{result_df['Default probability'].mean():.1%}")
    m3.metric("Manual review", int(result_df["Recommendation"].str.contains("manual review", case=False).sum()))
    st.dataframe(result_df.style.format({"Default probability": "{:.2%}"}), use_container_width=True, hide_index=True)
    st.download_button("Download scored rows and explanations", result_df.to_csv(index=False), "credisense_scored_explanations.csv", "text/csv")

    risk_fig = px.histogram(result_df, x="Default probability", color="Recommendation", nbins=24,
                            title="Current batch risk distribution", hover_data=["Applicant"], barmode="overlay")
    risk_fig.update_layout(height=380, xaxis_tickformat=".0%")
    st.plotly_chart(risk_fig, use_container_width=True)

    st.subheader("Explore one applicant")
    selected = st.selectbox("Applicant", result_df["Applicant"].tolist())
    idx = result_df.index[result_df["Applicant"] == selected][0]
    contributions = pd.DataFrame({"Feature": [n.replace("num__", "").replace("cat__", "") for n in names],
                                  "SHAP contribution": sv[idx]}).sort_values("SHAP contribution", key=lambda s: s.abs()).tail(15)
    contributions["Direction"] = np.where(contributions["SHAP contribution"] >= 0, "Raises model score", "Lowers model score")
    driver_fig = px.bar(contributions, x="SHAP contribution", y="Feature", color="Direction", orientation="h",
                        color_discrete_map={"Raises model score": "#d95f59", "Lowers model score": "#159a8c"},
                        title=f"Top model contributions · applicant {selected}", hover_data={"SHAP contribution": ":.5f"})
    driver_fig.add_vline(x=0, line_color="#667085")
    driver_fig.update_layout(height=480, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(driver_fig, use_container_width=True)
    st.info(result_df.loc[idx, "Reasoning (top SHAP drivers)"] + "\n\nRecommendation: " + result_df.loc[idx, "Recommendation"] + ". " + result_df.loc[idx, "Review note"])

    numeric_names = [n for n in names if n.startswith("num__")]
    if len(numeric_names) > 1:
        st.subheader("Batch feature interaction view")
        selected_feature = st.selectbox("Feature for dependence view", numeric_names, format_func=lambda n: n.replace("num__", ""))
        feature_index = names.index(selected_feature)
        shap_column = f"SHAP · {selected_feature.replace('num__', '')}"
        dependence = pd.DataFrame({"Feature value": dense_values[:, feature_index], shap_column: sv[:, feature_index],
                                   "Applicant": result_df["Applicant"], "Default probability": result_df["Default probability"]})
        fig_dep = px.scatter(dependence, x="Feature value", y=shap_column, color="Default probability",
                             hover_name="Applicant", color_continuous_scale="RdYlGn_r",
                             title="Feature value and its contribution across the current batch")
        st.plotly_chart(fig_dep, use_container_width=True)

    if "Risk_Flag" in raw.columns:
        st.markdown("---")
        st.subheader("Fairness screening (requires observed outcome labels)")
        try:
            fair_rows = create_features(raw.copy())
            for col in expected_cols:
                if col not in fair_rows.columns:
                    fair_rows[col] = 0
            metrics = subgroup_metrics(model, fair_rows, group_cols=[c for c in ["age_group", "House_Ownership", "Married/Single"] if c in fair_rows.columns])
            if metrics.empty:
                st.info("Not enough labelled records per subgroup to calculate screening metrics.")
            else:
                fair_fig = px.bar(metrics, x="group_val", y="approval_rate", color="group_col", barmode="group",
                                  hover_data=["n", "AUC", "F1", "default_rate"], title="Observed-label approval rate by subgroup")
                fair_fig.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="Screening reference")
                st.plotly_chart(fair_fig, use_container_width=True)
                st.dataframe(disparate_impact_ratio(metrics), use_container_width=True, hide_index=True)
                st.caption("Exploratory screening only; the 0.8 reference is not a legal determination or proof of fairness.")
        except Exception as exc:
            st.warning(f"Fairness screening unavailable: {exc}")
