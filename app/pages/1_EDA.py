import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))
data_path = BASE_DIR / "data" / "loan_cleaned.csv"

from src.live_data import get_macro_indicators, get_rbi_repo_rate

st.set_page_config(layout="wide")
st.title("📊 Data Analysis Dashboard")

tabs = st.tabs(["🗂️ Dataset EDA", "🌐 Live Macro Indicators"])

# ── TAB 1: Dataset EDA ─────────────────────────────────────────────────────────
with tabs[0]:
    try:
        df = pd.read_csv(data_path, sep=None, engine="python")
        df.columns = df.columns.str.strip()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    if df.empty:
        st.warning("Dataset is empty")
        st.stop()

    df["Risk_Flag"] = df["Risk_Flag"].astype(str)
    st.success(f"✅ Data Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Default Rate", f"{(df['Risk_Flag'] == '1').mean()*100:.2f}%")
    c3.metric("Avg Income", f"{df['Income'].mean():.3f}")
    c4.metric("Avg Experience", f"{df['Experience'].mean():.3f}")
    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("🔹 Risk Distribution")
        fig1 = px.pie(df, names="Risk_Flag", color="Risk_Flag",
                      color_discrete_map={"0": "#28a745", "1": "#dc3545"},
                      title="Safe vs Default")
        st.plotly_chart(fig1, use_container_width=True)

    with col_r:
        st.subheader("🔹 House Ownership vs Risk")
        if "House_Ownership" in df.columns:
            grp = df.groupby(["House_Ownership", "Risk_Flag"]).size().reset_index(name="count")
            fig_own = px.bar(grp, x="House_Ownership", y="count", color="Risk_Flag",
                             barmode="group", color_discrete_map={"0": "#28a745", "1": "#dc3545"},
                             title="Ownership Type vs Default")
            st.plotly_chart(fig_own, use_container_width=True)

    st.markdown("---")
    df_sample = df.sample(n=10000, random_state=42)

    col2_l, col2_r = st.columns(2)
    with col2_l:
        st.subheader("🔹 Income Distribution by Risk")
        fig2 = px.box(df_sample, x="Risk_Flag", y="Income", color="Risk_Flag",
                      color_discrete_map={"0": "#28a745", "1": "#dc3545"},
                      title="Income vs Default (10k sample)")
        st.plotly_chart(fig2, use_container_width=True)

    with col2_r:
        st.subheader("🔹 Experience vs Risk")
        fig3 = px.box(df_sample, x="Risk_Flag", y="Experience", color="Risk_Flag",
                      color_discrete_map={"0": "#28a745", "1": "#dc3545"},
                      title="Experience vs Default (10k sample)")
        st.plotly_chart(fig3, use_container_width=True)

    if "Profession" in df.columns:
        st.markdown("---")
        st.subheader("🔹 Top Professions by Default Rate")
        prof = (df[df["Risk_Flag"] == "1"].groupby("Profession").size() /
                df.groupby("Profession").size()).dropna().sort_values(ascending=False).head(15)
        fig_prof = px.bar(x=prof.index, y=prof.values, labels={"x": "Profession", "y": "Default Rate"},
                          title="Default Rate by Profession (Top 15)", color=prof.values,
                          color_continuous_scale="Reds")
        st.plotly_chart(fig_prof, use_container_width=True)

# ── TAB 2: Live Macro Indicators ───────────────────────────────────────────────
with tabs[1]:
    st.subheader("🌐 India Macroeconomic Context")
    st.caption("Live data from World Bank Open Data API · Cached hourly")

    with st.spinner("Fetching live indicators..."):
        macro = get_macro_indicators()
        repo_rates = get_rbi_repo_rate()

    # Latest values KPI row
    kpi_cols = st.columns(len(macro))
    for i, (name, df_m) in enumerate(macro.items()):
        if not df_m.empty:
            latest = df_m.iloc[-1]
            prev = df_m.iloc[-2] if len(df_m) > 1 else latest
            delta = round(latest["value"] - prev["value"], 2)
            kpi_cols[i].metric(name, f"{latest['value']:.2f}%",
                                delta=f"{delta:+.2f}% vs prior year")

    st.markdown("---")
    st.caption("💡 Higher unemployment & NPA ratios historically correlate with increased loan defaults.")

    chart_cols = st.columns(2)
    items = list(macro.items())

    for i, (name, df_m) in enumerate(items):
        with chart_cols[i % 2]:
            if not df_m.empty:
                fig = px.line(df_m, x="year", y="value", title=f"India: {name}",
                              markers=True, labels={"value": name, "year": "Year"})
                fig.update_traces(line_color="#1f77b4")
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🏦 RBI Repo Rate History")
    st.caption("Higher repo rate → costlier credit → higher default risk")
    df_repo = pd.DataFrame(repo_rates)
    fig_repo = px.line(df_repo, x="date", y="rate", markers=True,
                       title="RBI Repo Rate (%)", labels={"rate": "Rate (%)", "date": "Date"})
    fig_repo.update_traces(line_color="#e74c3c", line_width=2)
    st.plotly_chart(fig_repo, use_container_width=True)
