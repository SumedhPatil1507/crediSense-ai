import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]

model_path = BASE_DIR / "models" / "model.pkl"
columns_path = BASE_DIR / "models" / "columns.json"

model = joblib.load(model_path)

# Fallback columns 
default_cols = [
    "Income","Age","Experience",
    "CURRENT_JOB_YRS","CURRENT_HOUSE_YRS",
    "House_Ownership","Married/Single",
    "Car_Ownership","Profession",
    "CITY","STATE","age_group",
    "income_per_job_year","total_stability","experience_ratio"
]

# Load safely
try:
    import json
    cols = json.load(open(columns_path))
except:
    st.warning("⚠️ Using default columns")
    cols = default_cols

st.title("🔍 Loan Risk Prediction")

# Inputs
income = st.slider("Income", 0.0, 1.0)
age = st.slider("Age", 0.0, 1.0)
exp = st.slider("Experience", 0.0, 1.0)

if st.button("Predict"):

    df = pd.DataFrame(columns=cols)
    df.loc[0] = 0

    # Fill inputs
    df["Income"] = income
    df["Age"] = age
    df["Experience"] = exp
    df["CURRENT_JOB_YRS"] = 2
    df["CURRENT_HOUSE_YRS"] = 3
    df["House_Ownership"] = "owned"
    df["Married/Single"] = "single"
    df["Car_Ownership"] = "no"
    df["Profession"] = "Engineer"
    df["CITY"] = "Mumbai"
    df["STATE"] = "Maharashtra"
    df["age_group"] = "Middle"

    # Feature engineering (IMPORTANT)
    df["income_per_job_year"] = df["Income"] / (df["CURRENT_JOB_YRS"] + 1)
    df["total_stability"] = df["CURRENT_JOB_YRS"] + df["CURRENT_HOUSE_YRS"]
    df["experience_ratio"] = df["Experience"] / (df["Age"] + 1)

    prob = model.predict_proba(df)[0][1]

    st.metric("Risk Probability", f"{prob:.2%}")

    if prob > 0.5:
        st.error("⚠️ High Risk")
    else:
        st.success("✅ Safe")

    # Business logic
    profit = (1 - prob)*20000 - prob*100000
    st.metric("Expected Profit", f"₹{profit:,.0f}")