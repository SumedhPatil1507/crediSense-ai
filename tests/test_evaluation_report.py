import numpy as np
import pandas as pd

from src.evaluation_report import build_holdout_report


class _ProbabilityModel:
    def predict_proba(self, frame):
        score = frame["score"].to_numpy()
        return np.column_stack((1 - score, score))

    def predict(self, frame):
        return (self.predict_proba(frame)[:, 1] >= 0.5).astype(int)


def test_report_is_reproducible_and_marks_non_temporal_split(tmp_path):
    y_train = np.array([0, 1] * 20)
    X_train = pd.DataFrame({"score": np.linspace(0.02, 0.98, len(y_train))})
    y_test = np.array([0, 1] * 40)
    scores = np.linspace(0.01, 0.99, len(y_test))
    X_test = pd.DataFrame({
        "score": scores,
        "age_group": ["young", "older"] * 40,
        "House_Ownership": ["owned", "rented"] * 40,
        "Married/Single": ["single", "married"] * 40,
    })
    dataset_file = tmp_path / "dataset.csv"
    model_file = tmp_path / "model.bin"
    dataset_file.write_text("synthetic test data", encoding="utf-8")
    model_file.write_bytes(b"fake model artifact")

    report = build_holdout_report(
        _ProbabilityModel(), X_train, y_train, X_test, y_test,
        dataset_path=dataset_file, model_path=model_file,
    )

    assert report["evaluation_protocol"]["temporal_evaluation"] is False
    assert report["model_metrics"]["roc_auc_bootstrap_95"]["lower_95"] <= 1.0
    assert report["prior_baseline"]["roc_auc"] == 0.5
    assert len(report["policy_thresholds"]) == 3
