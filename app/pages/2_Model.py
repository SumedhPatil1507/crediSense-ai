import streamlit as st
import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

st.title("🔍 Loan Risk Prediction")

# Layout
col1, col2 = st.columns(2)

with col1:
    income = st.slider("Income", 0.0, 1.0)
    age = st.slider("Age", 0.0, 1.0)

with col2:
    exp = st.slider("Experience", 0.0, 1.0)
    house = st.selectbox("House Ownership", ["owned", "rented", "norent_noown"])

if st.button("🚀 Predict Risk"):

    df = pd.DataFrame([{
        "Income": income,
        "Age": age,
        "Experience": exp,
        "House_Ownership": house
    }])

    prob = model.predict_proba(df)[0][1]
    pred = model.predict(df)[0]

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric("Risk Probability", f"{prob:.2%}")
    
    decision = "Approve" if prob < 0.3 else "Review" if prob < 0.6 else "Reject"
    col2.metric("Decision", decision)

    # Progress bar
    col3.progress(int(prob * 100))

    # Color feedback
    if pred:
        st.error("⚠️ High Risk Applicant")
    else:
        st.success("✅ Safe Applicant")

    # 💰 Business Logic
    loan = 100000
    profit = (1 - prob)*20000 - prob*loan
    st.metric("Expected Profit/Loss", f"₹{profit:,.0f}")