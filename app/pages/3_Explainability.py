import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.feature_engineering import create_features

model = joblib.load(BASE_DIR / "models/model.pkl")
pre = model.named_steps['preprocessor']
mod = model.named_steps['model']

# Use the exact columns the preprocessor was fitted on
expected_cols = list(pre.feature_names_in_)

st.set_page_config(layout="wide")
st.title("🧠 Explainability")

file = st.file_uploader("Upload CSV")

if file:
    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()

        # Drop target if present
        if "Risk_Flag" in df.columns:
            df = df.drop(columns=["Risk_Flag"])

        # Apply feature engineering
        df = create_features(df)

        # Add Id if missing
        if "Id" not in df.columns:
            df["Id"] = 0

        # Align to exact columns the preprocessor expects
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]

        sample = df.head(50)

        X = pre.transform(sample)

        explainer = shap.TreeExplainer(mod)
        shap_values = explainer.shap_values(X)

        sv = shap_values[1] if isinstance(shap_values, list) else shap_values

        shap.summary_plot(sv, X, show=False)
        st.pyplot(plt.gcf())
        plt.clf()

    except Exception as e:
        st.error(f"Error: {e}")
