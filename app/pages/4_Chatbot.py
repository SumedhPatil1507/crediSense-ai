import streamlit as st
import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

st.title("🤖 Chatbot")

text = st.text_input("Enter: income,age,experience")

if text:
    vals = list(map(float, text.split(',')))

    df = pd.DataFrame([{
        "Income": vals[0],
        "Age": vals[1],
        "Experience": vals[2]
    }])

    prob = model.predict_proba(df)[0][1]

    if prob > 0.5:
        st.error("High Risk")
    else:
        st.success("Safe")