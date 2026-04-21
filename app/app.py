import streamlit as st

st.set_page_config(page_title="CrediSense AI", layout="wide", page_icon="💳")

st.title("💳 CrediSense AI")
st.markdown("### AI-Powered Credit Risk Scoring System")
st.markdown("---")

c1, c2, c3, c4, c5 = st.columns(5)
c1.info("📊 **EDA**\nDataset + live macro indicators")
c2.success("💳 **Model**\nPredict · Metrics · Compare · Threshold")
c3.warning("🧠 **Explainability**\nSHAP summary · Waterfall · Dependence")
c4.error("🤖 **Chatbot**\nRisk assistant + Q&A knowledge base")
c5.info("📋 **Logs**\nUsage logs · Feedback · Analysis")

st.markdown("---")
col_l, col_r = st.columns(2)
with col_l:
    st.markdown("""
    **What this system does:**
    - Predicts loan default probability using LightGBM
    - Explains predictions with SHAP values
    - Tracks analyst feedback for model improvement
    - Shows live India macroeconomic context
    """)
with col_r:
    st.markdown("""
    **Data & Model:**
    - 252,000 loan records · 12% default rate
    - LightGBM · ROC-AUC ~0.97 · Gini ~0.94
    - [Kaggle Dataset](https://www.kaggle.com/datasets/subhamjain/loan-prediction-based-on-customer-behavior) ·
      [LightGBM Paper](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
    """)

st.sidebar.success("Select a page above 👆")
