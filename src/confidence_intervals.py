"""Bootstrap confidence intervals for model predictions."""
import numpy as np
import warnings


def bootstrap_ci(model, df_input, n_bootstrap: int = 200, ci: float = 0.95):
    """
    Estimate prediction confidence interval via bootstrap resampling
    of the input features (simulates input uncertainty).

    Returns (point_estimate, lower_bound, upper_bound).
    """
    rng = np.random.default_rng(42)
    preds = []

    import pandas as pd
    num_cols = df_input.select_dtypes(include=[float, int]).columns.tolist()
    base_arr = df_input[num_cols].values.astype(float)

    for _ in range(n_bootstrap):
        noisy_df = df_input.copy()
        noise = rng.normal(0, 0.01, size=base_arr.shape)
        noisy_vals = np.clip(base_arr + noise, 0, 1)
        noisy_df[num_cols] = noisy_vals
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = model.predict_proba(noisy_df)[0][1]
        preds.append(p)

    preds = np.array(preds)
    alpha = (1 - ci) / 2
    lower = float(np.quantile(preds, alpha))
    upper = float(np.quantile(preds, 1 - alpha))
    point = float(np.mean(preds))
    return point, lower, upper


def interpret_ci(lower: float, upper: float) -> str:
    width = upper - lower
    if width < 0.05:
        return "Very tight — model is highly certain"
    elif width < 0.15:
        return "Moderate — reasonable certainty"
    else:
        return "Wide — borderline case, manual review recommended"
