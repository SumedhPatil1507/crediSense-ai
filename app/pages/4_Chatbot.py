import streamlit as st
import joblib
import pandas as pd

st.title("🤖 Credit Risk Assistant")

model = joblib.load("models/model.pkl")

st.write("Ask about loan risk decisions!")

user_input = st.text_input("Enter applicant details (Income, Age, Experience):")

if user_input:
    try:
        vals = list(map(float, user_input.split(',')))

        df = pd.DataFrame([{
            "Income": vals[0],
            "Age": vals[1],
            "Experience": vals[2],
            "House_Ownership": "owned"
        }])

        prob = model.predict_proba(df)[0][1]

        if prob > 0.5:
            st.error(f"⚠️ High Risk ({prob:.2%})")
            st.write("Reason: Low stability or income-risk mismatch.")
        else:
            st.success(f"✅ Safe ({prob:.2%})")
            st.write("Reason: Stable profile with good income.")
    except:
        st.warning("Enter values like: 0.5,0.3,0.2")