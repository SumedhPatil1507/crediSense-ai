import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --- PATH ---
BASE_DIR = Path(__file__).resolve().parents[2]
data_path = BASE_DIR / "data" / "loan_cleaned.csv"

st.set_page_config(layout="wide")
st.title("📊 Data Analysis Dashboard")

# --- LOAD DATA ---
try:
    df = pd.read_csv(data_path, sep=None, engine="python")
    df.columns = df.columns.str.strip()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- CHECK DATA ---
if df.empty:
    st.warning("Dataset is empty")
    st.stop()

st.success(f"✅ Data Loaded: {df.shape}")

# --- KPI SECTION ---
col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(df))
col2.metric("Risk %", f"{df['Risk_Flag'].mean()*100:.2f}%")
col3.metric("Avg Income", f"{df['Income'].mean():.2f}")

st.markdown("---")

# --- PLOTS SECTION ---

# 1. Risk Distribution
st.subheader("🔹 Risk Distribution")

fig1 = px.pie(
    df,
    names="Risk_Flag",
    title="Risk vs Safe Distribution",
    color="Risk_Flag",
    color_discrete_map={0: "green", 1: "red"}
)

st.plotly_chart(fig1, use_container_width=True)

# 2. Income vs Risk
st.subheader("🔹 Income Distribution by Risk")

fig2 = px.box(
    df,
    x="Risk_Flag",
    y="Income",
    color="Risk_Flag",
    title="Income vs Risk"
)

st.plotly_chart(fig2, use_container_width=True)

# 3. Age Distribution
st.subheader("🔹 Age Distribution")

fig3 = px.histogram(
    df,
    x="Age",
    color="Risk_Flag",
    nbins=30,
    title="Age Distribution by Risk"
)

st.plotly_chart(fig3, use_container_width=True)

# 4. Experience vs Risk
st.subheader("🔹 Experience vs Risk")

fig4 = px.box(
    df,
    x="Risk_Flag",
    y="Experience",
    color="Risk_Flag",
    title="Experience vs Risk"
)

st.plotly_chart(fig4, use_container_width=True)