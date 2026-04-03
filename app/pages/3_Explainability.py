import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

model = joblib.load(BASE_DIR / "models/model.pkl")

st.set_page_config(layout="wide")
st.title("🧠 Explainability")

file = st.file_uploader("Upload CSV")

if file:
    df = pd.read_csv(file)

    pre = model.named_steps['preprocessor']
    mod = model.named_steps['model']

    X = pre.transform(df.head(50))

    explainer = shap.TreeExplainer(mod)
    shap_values = explainer.shap_values(X)

    # For binary classification, shap_values is a list; use class 1
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    fig, ax = plt.subplots()
    shap.summary_plot(sv, X, show=False)
    st.pyplot(plt.gcf())