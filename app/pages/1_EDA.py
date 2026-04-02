import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Data Insights Dashboard")

df = pd.read_csv("data/loan_cleaned.csv", sep=None, engine="python")

# Clean
df.columns = df.columns.str.strip()

# ---- KPI CARDS ----
col1, col2, col3 = st.columns(3)
col1.metric("Total Applicants", len(df))
col2.metric("Risk %", f"{df['Risk_Flag'].mean()*100:.2f}%")
col3.metric("Avg Income", f"{df['Income'].mean():.2f}")

st.markdown("---")

# ---- INTERACTIVE PLOTS ----

# Risk Distribution
fig1 = px.pie(df, names="Risk_Flag", title="Risk Distribution")
st.plotly_chart(fig1, use_container_width=True)

# Income vs Risk
fig2 = px.box(df, x="Risk_Flag", y="Income", color="Risk_Flag",
              title="Income Distribution by Risk")
st.plotly_chart(fig2, use_container_width=True)

# Age vs Risk
fig3 = px.histogram(df, x="Age", color="Risk_Flag",
                    title="Age Distribution by Risk",
                    barmode="overlay")
st.plotly_chart(fig3, use_container_width=True)