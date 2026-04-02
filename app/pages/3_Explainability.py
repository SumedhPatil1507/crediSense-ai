import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

model = joblib.load("models/model.pkl")

st.title("🧠 Explainability")

file = st.file_uploader("Upload CSV")

if file:
    df = pd.read_csv(file)

    pre = model.named_steps['preprocessor']
    mod = model.named_steps['model']

    X = pre.transform(df.head(50))

    explainer = shap.TreeExplainer(mod)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(plt)