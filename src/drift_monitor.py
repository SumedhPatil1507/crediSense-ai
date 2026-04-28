"""
PSI (Population Stability Index) and CSI (Characteristic Stability Index) tracking.
Detects data drift between training distribution and live predictions.
PSI < 0.1: No drift | 0.1-0.2: Moderate | > 0.2: Significant drift
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between expected (training) and actual (live) distributions."""
    expected = np.array(expected, dtype=float)
    actual   = np.array(actual,   dtype=float)

    breakpoints = np.linspace(0, 1, bins + 1)
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct   = np.histogram(actual,   bins=breakpoints)[0] / len(actual)

    # Avoid division by zero
    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct   = np.where(actual_pct   == 0, 1e-6, actual_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return round(float(psi), 4)


def interpret_psi(psi: float) -> tuple[str, str]:
    """Returns (status, message)."""
    if psi < 0.1:
        return "stable", "No significant drift detected"
    elif psi < 0.2:
        return "warning", "Moderate drift — monitor closely"
    else:
        return "critical", "Significant drift — consider retraining"


def compute_csi(training_df: pd.DataFrame, live_df: pd.DataFrame,
                features: list[str]) -> pd.DataFrame:
    """CSI per feature — which features are drifting most."""
    results = []
    for feat in features:
        if feat in training_df.columns and feat in live_df.columns:
            try:
                psi = compute_psi(
                    training_df[feat].dropna().values,
                    live_df[feat].dropna().values
                )
                status, msg = interpret_psi(psi)
                results.append({"feature": feat, "psi": psi, "status": status, "note": msg})
            except Exception:
                pass
    return pd.DataFrame(results).sort_values("psi", ascending=False)


def get_training_score_distribution() -> np.ndarray:
    """Load training risk scores from the saved dataset for PSI baseline."""
    import joblib, json, warnings
    model_path = BASE_DIR / "models" / "model.pkl"
    data_path  = BASE_DIR / "data"  / "loan_cleaned.csv"
    cols_path  = BASE_DIR / "models" / "columns.json"

    if not model_path.exists() or not data_path.exists():
        return np.array([])

    try:
        model = joblib.load(model_path)
        with open(cols_path) as f:
            cols = json.load(f)

        from src.data_loader import load_data
        from src.preprocessing import clean_data
        from src.feature_engineering import create_features
        from sklearn.model_selection import train_test_split

        df = load_data(str(data_path))
        df = clean_data(df)
        df = create_features(df)
        X  = df.drop(columns=["Risk_Flag"])
        _, X_test, _, _ = train_test_split(X, df["Risk_Flag"], test_size=0.2,
                                            stratify=df["Risk_Flag"], random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = model.predict_proba(X_test)[:, 1]
        return scores
    except Exception:
        return np.array([])
