"""Reproducible holdout report generation for model training releases."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.calibration import calibration_curve
from src.telemetry import traced_predict_proba
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_score, recall_score, roc_auc_score)

from src.evaluate import threshold_analysis
from src.fairness import subgroup_metrics


def _bootstrap_interval(metric, y_true, scores, *, seed=19, iterations=300):
    y = np.asarray(y_true)
    p = np.asarray(scores)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size < 2:
            continue
        estimates.append(metric(y[idx], p[idx]))
    if not estimates:
        return None
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {"lower_95": round(float(lower), 6), "upper_95": round(float(upper), 6)}


def build_holdout_report(model, X_train, y_train, X_test, y_test, *,
                         dataset_path, model_path, split_seed=42):
    """Evaluate one untouched split and include traceable hashes and caveats."""
    y = np.asarray(y_test, dtype=int)
    scores = np.asarray(traced_predict_proba(model, X_test, operation="evaluation_report")[:, 1], dtype=float)
    labels = (scores >= 0.5).astype(int)
    baseline = DummyClassifier(strategy="prior")
    baseline.fit(X_train, y_train)
    baseline_scores = traced_predict_proba(baseline, X_test, operation="evaluation_baseline")[:, 1]
    calibration_y, calibration_pred = calibration_curve(y, scores, n_bins=10, strategy="quantile")

    test_context = X_test.copy()
    test_context["Risk_Flag"] = y
    groups = subgroup_metrics(model, test_context)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_protocol": {
            "split_strategy": "stratified_random_holdout",
            "test_fraction": 0.2,
            "random_seed": split_seed,
            "temporal_evaluation": False,
            "temporal_limitation": "No verified decision-time field is configured; temporal generalization is untested.",
            "test_rows": int(len(y)),
            "test_default_rate": round(float(y.mean()), 6),
        },
        "artifacts": {
            "dataset_sha256": hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest(),
            "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        },
        "model_metrics": {
            "roc_auc": round(float(roc_auc_score(y, scores)), 6),
            "roc_auc_bootstrap_95": _bootstrap_interval(roc_auc_score, y, scores),
            "pr_auc_average_precision": round(float(average_precision_score(y, scores)), 6),
            "pr_auc_bootstrap_95": _bootstrap_interval(average_precision_score, y, scores),
            "brier_score": round(float(brier_score_loss(y, scores)), 6),
            "precision_at_0_5": round(float(precision_score(y, labels, zero_division=0)), 6),
            "recall_at_0_5": round(float(recall_score(y, labels, zero_division=0)), 6),
            "calibration_curve": [{"mean_predicted": round(float(x), 6), "fraction_positive": round(float(yv), 6)}
                                  for yv, x in zip(calibration_y, calibration_pred)],
        },
        "prior_baseline": {
            "roc_auc": round(float(roc_auc_score(y, baseline_scores)), 6),
            "pr_auc_average_precision": round(float(average_precision_score(y, baseline_scores)), 6),
        },
        "policy_thresholds": threshold_analysis(y, scores, thresholds=[0.30, 0.50, 0.60]),
        "exploratory_subgroups": json.loads(groups.to_json(orient="records")),
        "limitations": [
            "Random holdout is not a temporal or external validation.",
            "Subgroup results are exploratory and do not establish legal compliance or absence of harm.",
            "Bootstrap intervals represent sampling uncertainty on this test split only.",
        ],
    }
