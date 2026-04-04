from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    confusion_matrix, precision_recall_curve, roc_curve
)
import numpy as np

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "F1": round(f1_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_prob), 4),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "roc_curve": roc_curve(y_test, y_prob),
        "pr_curve": precision_recall_curve(y_test, y_prob),
        "y_prob": y_prob,
        "y_pred": y_pred,
        "y_test": y_test,
    }

def threshold_analysis(y_test, y_prob, thresholds=None):
    """Returns precision, recall, F1, approval_rate at each threshold."""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)
    results = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        approval_rate = (y_pred == 0).mean()
        results.append({
            "threshold": round(t, 2),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "approval_rate": round(approval_rate, 3),
        })
    return results
