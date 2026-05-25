"""
Fairness & Bias Analysis.
Checks model performance across demographic subgroups.
Uses age_group and house_ownership as proxy attributes.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def subgroup_metrics(model, df: pd.DataFrame, target_col: str = "Risk_Flag",
                     group_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Compute AUC, F1, approval_rate per subgroup.
    Returns a DataFrame with one row per subgroup value.
    """
    import warnings
    if group_cols is None:
        group_cols = ["age_group", "House_Ownership", "Married/Single"]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)

    rows = []
    for col in group_cols:
        if col not in df.columns:
            continue
        for val in df[col].unique():
            mask = df[col] == val
            if mask.sum() < 30:
                continue
            try:
                auc = roc_auc_score(y[mask], y_prob[mask])
                f1  = f1_score(y[mask], y_pred[mask])
                approval = (y_pred[mask] == 0).mean()
                default_rate = y[mask].mean()
                rows.append({
                    "group_col": col,
                    "group_val": str(val),
                    "n": int(mask.sum()),
                    "AUC": round(auc, 4),
                    "F1": round(f1, 4),
                    "approval_rate": round(approval, 4),
                    "default_rate": round(default_rate, 4),
                })
            except Exception:
                pass

    return pd.DataFrame(rows)


def disparate_impact_ratio(df_metrics: pd.DataFrame,
                            metric: str = "approval_rate") -> pd.DataFrame:
    """
    Disparate Impact Ratio = min_group_rate / max_group_rate.
    < 0.8 is considered discriminatory (4/5ths rule).
    """
    if df_metrics.empty or metric not in df_metrics.columns:
        return pd.DataFrame()

    rows = []
    for col in df_metrics["group_col"].unique():
        sub = df_metrics[df_metrics["group_col"] == col]
        max_val = sub[metric].max()
        min_val = sub[metric].min()
        dir_ratio = min_val / max_val if max_val > 0 else None
        rows.append({
            "group_col": col,
            "max_group": sub.loc[sub[metric].idxmax(), "group_val"],
            "min_group": sub.loc[sub[metric].idxmin(), "group_val"],
            f"max_{metric}": round(max_val, 4),
            f"min_{metric}": round(min_val, 4),
            "disparate_impact_ratio": round(dir_ratio, 4) if dir_ratio else None,
            "fair": "Yes" if dir_ratio and dir_ratio >= 0.8 else "No",
        })
    return pd.DataFrame(rows)
