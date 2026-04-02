import streamlit as st
import joblib
import pandas as pd
import json

model = joblib.load("models/model.pkl")
cols = json.load(open("models/columns.json"))

st.title("🔍 Prediction")

income = st.slider("Income", 0.0, 1.0)
age = st.slider("Age", 0.0, 1.0)
exp = st.slider("Experience", 0.0, 1.0)

if st.button("Predict"):

    df = pd.DataFrame(columns=cols)
    df.loc[0] = 0

    df["Income"] = income
    df["Age"] = age
    df["Experience"] = exp

    prob = model.predict_proba(df)[0][1]

    st.metric("Risk", f"{prob:.2%}")