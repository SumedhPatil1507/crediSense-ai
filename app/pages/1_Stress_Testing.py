import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.stress_test import run_macro_stress_test
from src.live_data import get_macro_indicators

st.set_page_config(layout="wide")
st.title("🌪️ Macroeconomic Stress Testing")
st.markdown("Simulate portfolio-level shocks to evaluate model robustness and expected default rates.")

data_path = BASE_DIR / "data" / "loan_cleaned.csv"

@st.cache_data
def load_sample_data():
    df = pd.read_csv(data_path, sep=None, engine="python")
    df.columns = df.columns.str.strip()
    return df.sample(n=2000, random_state=42)

try:
    df_sample = load_sample_data()
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

st.sidebar.header("Shock Parameters")
income_mult = st.sidebar.slider("Income Multiplier", 0.5, 1.5, 1.0, 0.05, 
                                help="E.g. 0.8 means a 20% drop in income across the portfolio.")
exp_shock = st.sidebar.slider("Experience Multiplier", 0.5, 1.5, 1.0, 0.05)
job_shock = st.sidebar.slider("Job Tenure Multiplier", 0.5, 1.5, 1.0, 0.05)

if st.sidebar.button("Run Stress Test", type="primary"):
    with st.spinner("Simulating shocks..."):
        try:
            results = run_macro_stress_test(df_sample, income_mult, exp_shock, job_shock)
            
            # Key Metrics
            baseline_default_rate = (results["baseline_prob"] > 0.5).mean()
            shocked_default_rate = (results["shocked_prob"] > 0.5).mean()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Baseline Default Rate", f"{baseline_default_rate*100:.1f}%")
            col2.metric("Shocked Default Rate", f"{shocked_default_rate*100:.1f}%", 
                        delta=f"{(shocked_default_rate - baseline_default_rate)*100:.1f}%", delta_color="inverse")
            col3.metric("Avg PD Delta", f"{results['delta_prob'].mean()*100:+.2f} bps", delta_color="inverse")
            
            st.markdown("---")
            
            # Visualizations
            st.subheader("Probability of Default (PD) Distribution Shift")
            
            fig = px.histogram(results, x=["baseline_prob", "shocked_prob"], 
                               barmode="overlay", 
                               labels={"value": "Probability of Default", "variable": "Scenario"},
                               title="Shift in PD Distribution")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Delta Analysis")
            fig_delta = px.box(results, y="delta_prob", 
                               title="Distribution of PD Deltas across Portfolio")
            fig_delta.update_layout(height=400)
            st.plotly_chart(fig_delta, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error running stress test: {e}")

st.markdown("---")
st.subheader("Live Macro Indicators Tracker")
with st.spinner("Fetching live indicators..."):
    macro = get_macro_indicators()

kpi_cols = st.columns(len(macro))
for i, (name, df_m) in enumerate(macro.items()):
    if not df_m.empty:
        latest = df_m.iloc[-1]
        prev = df_m.iloc[-2] if len(df_m) > 1 else latest
        delta = round(float(latest["value"]) - float(prev["value"]), 2)
        kpi_cols[i].metric(name.split("(")[0].strip(),
                            f"{latest['value']:.2f}%",
                            delta=f"{delta:+.2f}%")
