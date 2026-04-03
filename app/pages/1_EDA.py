import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Data Dashboard")

df = pd.read_csv("data/loan_cleaned.csv", sep=None, engine="python")
df.columns = df.columns.str.strip()

# KPI
col1, col2 = st.columns(2)
col1.metric("Total Rows", len(df))
col2.metric("Risk %", f"{df['Risk_Flag'].mean()*100:.2f}%")

# Pie chart
fig1 = px.pie(df, names="Risk_Flag", title="Risk Distribution")
st.plotly_chart(fig1, use_container_width=True)

# Box plot
fig2 = px.box(df, x="Risk_Flag", y="Income", title="Income vs Risk")
st.plotly_chart(fig2, use_container_width=True)

# Histogram
fig3 = px.histogram(df, x="Age", color="Risk_Flag", title="Age Distribution")
st.plotly_chart(fig3, use_container_width=True)