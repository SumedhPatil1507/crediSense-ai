"""
Algorithmic Fairness & Bias Auditing Module.
Integrates Fairlearn to assess Demographic Parity and Disparate Impact.
"""
import pandas as pd
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio

def run_fairness_audit(y_true: pd.Series, y_pred: pd.Series, sensitive_features: pd.Series) -> dict:
    """
    Computes fairness metrics for a given sensitive feature (e.g. Gender, Age Group, State).
    y_true: Ground truth labels
    y_pred: Model predicted labels (0 or 1)
    sensitive_features: Series containing the protected attribute
    """
    try:
        dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
        dp_ratio = demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive_features)
        
        # Disparate Impact Ratio is typically analogous to Demographic Parity Ratio in binary classification
        return {
            "demographic_parity_difference": dp_diff,
            "disparate_impact_ratio": dp_ratio,
            "status": "Warning: High Disparity" if dp_ratio < 0.8 else "Acceptable"
        }
    except Exception as e:
        return {
            "error": str(e)
        }
