from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    confusion_matrix, precision_recall_curve, roc_curve,
    average_precision_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
import numpy as np


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = round(roc_auc_score(y_test, y_prob), 4)
    pr_auc = round(average_precision_score(y_test, y_prob), 4)
    gini = round(2 * auc - 1, 4)
    ks = round(_ks_statistic(y_test, y_prob), 4)
    brier = round(brier_score_loss(y_test, y_prob), 4)

    return {
        "F1": round(f1_score(y_test, y_pred), 4),
        "AUC": auc,
        "PR_AUC": pr_auc,
        "Gini": gini,
        "KS": ks,
        "Brier": brier,
        "report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "roc_curve": roc_curve(y_test, y_prob),
        "pr_curve": precision_recall_curve(y_test, y_prob),
        "calibration": calibration_curve(y_test, y_prob, n_bins=10),
        "y_prob": y_prob,
        "y_pred": y_pred,
        "y_test": np.array(y_test),
    }


def _ks_statistic(y_test, y_prob):
    """Kolmogorov-Smirnov statistic — max separation between TPR and FPR curves."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    return float(np.max(tpr - fpr))


def threshold_analysis(y_test, y_prob, thresholds=None):
    """Precision, recall, F1, approval_rate at each threshold."""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)
    y_test = np.array(y_test)
    results = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        results.append({
            "threshold": round(float(t), 2),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "approval_rate": round(float((y_pred == 0).mean()), 3),
        })
    return results


def compare_models(models_dict, X_test, y_test):
    """
    models_dict: {"Model Name": fitted_pipeline, ...}
    Returns a DataFrame-ready list of dicts with metrics per model.
    """
    rows = []
    for name, mdl in models_dict.items():
        y_pred = mdl.predict(X_test)
        y_prob = mdl.predict_proba(X_test)[:, 1]
        auc = round(roc_auc_score(y_test, y_prob), 4)
        rows.append({
            "Model": name,
            "ROC-AUC": auc,
            "PR-AUC": round(average_precision_score(y_test, y_prob), 4),
            "Gini": round(2 * auc - 1, 4),
            "KS": round(_ks_statistic(y_test, y_prob), 4),
            "F1": round(f1_score(y_test, y_pred), 4),
            "Brier": round(brier_score_loss(y_test, y_prob), 4),
        })
    return sorted(rows, key=lambda x: x["ROC-AUC"], reverse=True)
