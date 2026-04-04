import streamlit as st

st.set_page_config(page_title="CrediSense AI", layout="wide", page_icon="💳")

st.title("💳 CrediSense AI")
st.markdown("### AI-Powered Credit Risk Scoring System")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
col1.info("📊 **EDA**\nExplore the dataset and risk distributions")
col2.success("💳 **Model**\nPredict risk + view performance metrics")
col3.warning("🧠 **Explainability**\nSHAP feature importance analysis")
col4.error("🤖 **Chatbot**\nAsk the AI risk assistant")

st.markdown("---")
st.markdown(
    "Navigate using the sidebar. "
    "Upload `loan_cleaned.csv` in the Model and Explainability pages for live evaluation."
)
st.sidebar.success("Select a page above 👆")
