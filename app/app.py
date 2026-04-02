import streamlit as st

st.set_page_config(
    page_title="CrediSense AI",
    page_icon="💳",
    layout="wide"
)

# Sidebar branding
st.sidebar.title("💳 CrediSense AI")
st.sidebar.markdown("AI Credit Risk Platform")

st.sidebar.markdown("---")
st.sidebar.info("Built for FinTech Decision Systems")

# Main header
st.title("💳 CrediSense AI")
st.markdown("### AI-powered Credit Risk Scoring & Decision System")

# KPI placeholders (can connect later)
col1, col2, col3 = st.columns(3)

col1.metric("Model Type", "LightGBM")
col2.metric("Use Case", "Loan Approval")
col3.metric("Status", "Production Ready")

st.markdown("---")
st.success("Use sidebar to navigate between modules 👈")