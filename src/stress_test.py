"""
Macro Stress Testing Module.
Simulates portfolio-level macroeconomic shocks and their impact on expected default rates.
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from src.feature_engineering import create_features

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

def run_macro_stress_test(df_original: pd.DataFrame, 
                          income_multiplier: float = 1.0, 
                          experience_shock: float = 1.0,
                          job_yrs_shock: float = 1.0) -> pd.DataFrame:
    """
    Applies macroeconomic shocks to the dataset, re-calculates features,
    and returns the updated default probabilities.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError("model.pkl not found. Please train the model first.")
        
    model = joblib.load(MODEL_PATH)
    
    # Baseline predictions (if not already present)
    df_baseline = df_original.copy()
    df_baseline = create_features(df_baseline)
    X_baseline = df_baseline.drop(columns=["Risk_Flag", "Id"], errors="ignore")
    baseline_probs = model.predict_proba(X_baseline)[:, 1]
    
    # Apply Shocks
    df_shock = df_original.copy()
    df_shock["Income"] = df_shock["Income"] * income_multiplier
    df_shock["Experience"] = (df_shock["Experience"] * experience_shock).clip(lower=0)
    df_shock["CURRENT_JOB_YRS"] = (df_shock["CURRENT_JOB_YRS"] * job_yrs_shock).clip(lower=0)
    
    # Re-engineer features
    df_shock = create_features(df_shock)
    
    # Predict with shocked features
    X_shock = df_shock.drop(columns=["Risk_Flag", "Id"], errors="ignore")
    shock_probs = model.predict_proba(X_shock)[:, 1]
    
    # Results DataFrame
    results = pd.DataFrame({
        "baseline_prob": baseline_probs,
        "shocked_prob": shock_probs,
        "delta_prob": shock_probs - baseline_probs
    })
    
    return results
