import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.title("🧠 Model Explainability")

model = joblib.load("models/model.pkl")

file = st.file_uploader("Upload Sample Data (small CSV)")

if file:
    df = pd.read_csv(file)

    pre = model.named_steps['preprocessor']
    mod = model.named_steps['model']

    X = pre.transform(df.head(50))  # 🔥 LIMIT (important for cloud)

    explainer = shap.TreeExplainer(mod)
    shap_values = explainer.shap_values(X)

    st.subheader("Feature Importance")
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(plt)