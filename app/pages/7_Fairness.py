import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.fairness import run_fairness_audit
from src.feature_engineering import create_features

st.set_page_config(layout="wide")
st.title("⚖️ Algorithmic Fairness & Bias Auditing")
st.markdown("Monitor disparate impact and demographic parity across protected attributes.")

data_path = BASE_DIR / "data" / "loan_cleaned.csv"
model_path = BASE_DIR / "models" / "model.pkl"

@st.cache_data
def load_data_and_predict():
    df = pd.read_csv(data_path, sep=None, engine="python")
    df.columns = df.columns.str.strip()
    df = df.sample(n=3000, random_state=42)
    
    if not model_path.exists():
        st.error("Model not found.")
        return df, None
        
    model = joblib.load(model_path)
    df_feat = create_features(df.copy())
    X = df_feat.drop(columns=["Risk_Flag", "Id"], errors="ignore")
    y_true = df_feat["Risk_Flag"].astype(int)
    
    y_pred = model.predict(X)
    return df_feat, y_true, y_pred

with st.spinner("Analyzing fairness metrics..."):
    df, y_true, y_pred = load_data_and_predict()

if y_pred is not None:
    st.sidebar.header("Audit Configuration")
    
    # We will derive some sensitive features for demonstration if they don't exist
    if "age_group" not in df.columns:
        df["age_group"] = pd.qcut(df["Age"], q=3, labels=["Young", "Middle", "Senior"])
        
    sensitive_col = st.sidebar.selectbox("Select Sensitive Attribute", 
                                         options=["age_group", "Married/Single", "House_Ownership"])
    
    if st.sidebar.button("Run Audit", type="primary"):
        st.subheader(f"Auditing based on: **{sensitive_col}**")
        
        results = run_fairness_audit(y_true, pd.Series(y_pred), df[sensitive_col])
        
        if "error" in results:
            st.error(f"Error computing fairness: {results['error']}")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Demographic Parity Difference", f"{results['demographic_parity_difference']:.3f}",
                        help="Difference in selection rates between groups. Closer to 0 is better.")
            col2.metric("Disparate Impact Ratio", f"{results['disparate_impact_ratio']:.3f}",
                        help="Ratio of selection rates. Values < 0.8 indicate potential bias (80% rule).")
            
            status_color = "normal" if results['status'] == "Acceptable" else "inverse"
            col3.metric("Audit Status", results['status'])
            
            if results['disparate_impact_ratio'] < 0.8:
                st.warning("⚠️ **Potential Bias Detected:** The Disparate Impact Ratio falls below the acceptable 0.8 threshold. Consider mitigation strategies such as re-weighting or adversarial debiasing.")
            else:
                st.success("✅ **Fairness Acceptable:** Metrics fall within acceptable regulatory thresholds.")
                
            st.markdown("---")
            st.subheader("Group Selection Rates")
            
            # Compute selection rate manually for plotting
            df_eval = pd.DataFrame({
                "Group": df[sensitive_col],
                "Predicted_Risk": y_pred
            })
            
            rates = df_eval.groupby("Group")["Predicted_Risk"].mean().reset_index()
            rates.rename(columns={"Predicted_Risk": "Selection Rate (Predicted Default)"}, inplace=True)
            
            import plotly.express as px
            fig = px.bar(rates, x="Group", y="Selection Rate (Predicted Default)", 
                         color="Group", title=f"Predicted Default Rates by {sensitive_col}")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
