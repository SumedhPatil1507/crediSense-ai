"""Unit tests for the ML pipeline and src modules."""
import sys
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model():
    import joblib
    return joblib.load(BASE_DIR / "models" / "model.pkl")


@pytest.fixture(scope="module")
def cols():
    with open(BASE_DIR / "models" / "columns.json") as f:
        return json.load(f)


@pytest.fixture
def sample_input(cols):
    from utils import build_full_input
    return build_full_input({"Income": 0.5, "Age": 0.3, "Experience": 0.2}, cols)


# ── Feature Engineering ────────────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_creates_expected_features(self):
        from src.feature_engineering import create_features
        df = pd.DataFrame([{
            "Income": 0.5, "Age": 0.3, "Experience": 0.2,
            "CURRENT_JOB_YRS": 2, "CURRENT_HOUSE_YRS": 3
        }])
        result = create_features(df)
        expected = ["income_per_job_year", "total_stability", "experience_ratio",
                    "income_per_experience", "stability_score", "income_stability",
                    "low_income_flag", "high_stability_flag"]
        for feat in expected:
            assert feat in result.columns, f"Missing feature: {feat}"

    def test_no_nulls_after_engineering(self):
        from src.feature_engineering import create_features
        df = pd.DataFrame([{
            "Income": 0.5, "Age": 0.3, "Experience": 0.2,
            "CURRENT_JOB_YRS": 2, "CURRENT_HOUSE_YRS": 3
        }])
        result = create_features(df)
        assert result.isnull().sum().sum() == 0

    def test_low_income_flag(self):
        from src.feature_engineering import create_features
        df = pd.DataFrame([
            {"Income": 0.1, "Age": 0.3, "Experience": 0.2, "CURRENT_JOB_YRS": 2, "CURRENT_HOUSE_YRS": 3},
            {"Income": 0.9, "Age": 0.3, "Experience": 0.2, "CURRENT_JOB_YRS": 2, "CURRENT_HOUSE_YRS": 3},
        ])
        result = create_features(df)
        assert result.iloc[0]["low_income_flag"] == 1
        assert result.iloc[1]["low_income_flag"] == 0


# ── Model Prediction ───────────────────────────────────────────────────────────

class TestModelPrediction:
    def test_predict_returns_probability(self, model, sample_input):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob = model.predict_proba(sample_input)[0][1]
        assert 0.0 <= prob <= 1.0

    def test_predict_shape(self, model, sample_input):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.predict_proba(sample_input)
        assert result.shape == (1, 2)
        assert abs(result[0].sum() - 1.0) < 1e-6

    def test_high_risk_profile(self, model, cols):
        from utils import build_full_input
        import warnings
        df = build_full_input({"Income": 0.05, "Age": 0.1, "Experience": 0.02}, cols)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob = model.predict_proba(df)[0][1]
        assert prob > 0.3, "Low income/experience should have elevated risk"

    def test_low_risk_profile(self, model, cols):
        from utils import build_full_input
        import warnings
        df = build_full_input({"Income": 0.9, "Age": 0.6, "Experience": 0.7}, cols)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob = model.predict_proba(df)[0][1]
        assert prob < 0.7, "High income/experience should have lower risk"


# ── Evaluation Metrics ─────────────────────────────────────────────────────────

class TestEvaluationMetrics:
    def test_evaluate_returns_all_keys(self, model):
        from src.evaluate import evaluate
        from sklearn.model_selection import train_test_split
        from src.data_loader import load_data
        from src.preprocessing import clean_data
        from src.feature_engineering import create_features

        df = load_data(str(BASE_DIR / "data" / "loan_cleaned.csv"))
        df = clean_data(df)
        df = create_features(df)
        X = df.drop(columns=["Risk_Flag"])
        y = df["Risk_Flag"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.05, stratify=y, random_state=42)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = evaluate(model, X_test, y_test)

        for key in ["F1", "AUC", "PR_AUC", "Gini", "KS", "Brier"]:
            assert key in metrics, f"Missing metric: {key}"
        assert 0.8 < metrics["AUC"] < 1.0, "AUC should be > 0.8"
        assert metrics["Gini"] == round(2 * metrics["AUC"] - 1, 4)

    def test_threshold_analysis_returns_list(self, model):
        from src.evaluate import threshold_analysis
        y_test = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 0])
        y_prob = np.array([0.1, 0.8, 0.2, 0.9, 0.15, 0.75, 0.3, 0.4, 0.85, 0.05])
        results = threshold_analysis(y_test, y_prob)
        assert isinstance(results, list)
        assert len(results) > 0
        assert all("threshold" in r and "f1" in r for r in results)


# ── Confidence Intervals ───────────────────────────────────────────────────────

class TestConfidenceIntervals:
    def test_ci_bounds_valid(self, model, sample_input):
        from src.confidence_intervals import bootstrap_ci
        point, lower, upper = bootstrap_ci(model, sample_input, n_bootstrap=50)
        assert 0.0 <= lower <= point <= upper <= 1.0

    def test_ci_interpretation(self):
        from src.confidence_intervals import interpret_ci
        assert "tight" in interpret_ci(0.1, 0.12).lower() or "certain" in interpret_ci(0.1, 0.12).lower()
        assert "wide" in interpret_ci(0.1, 0.5).lower() or "borderline" in interpret_ci(0.1, 0.5).lower()


# ── Adverse Action ─────────────────────────────────────────────────────────────

class TestAdverseAction:
    def test_no_notice_for_low_risk(self):
        from src.adverse_action import generate_adverse_action
        result = generate_adverse_action(0.1, 0.8, 0.5, 0.6)
        assert result["required"] is False

    def test_notice_required_for_high_risk(self):
        from src.adverse_action import generate_adverse_action
        result = generate_adverse_action(0.85, 0.1, 0.2, 0.05)
        assert result["required"] is True
        assert len(result["reasons"]) > 0
        assert "notice" in result

    def test_notice_contains_citation(self):
        from src.adverse_action import generate_adverse_action
        result = generate_adverse_action(0.85, 0.1, 0.2, 0.05)
        assert "ECOA" in result["citation"] or "FCRA" in result["citation"]


# ── Drift Monitor ──────────────────────────────────────────────────────────────

class TestDriftMonitor:
    def test_psi_identical_distributions(self):
        from src.drift_monitor import compute_psi
        data = np.random.uniform(0, 1, 1000)
        psi = compute_psi(data, data)
        assert psi < 0.05

    def test_psi_different_distributions(self):
        from src.drift_monitor import compute_psi
        expected = np.random.uniform(0, 0.3, 1000)
        actual   = np.random.uniform(0.7, 1.0, 1000)
        psi = compute_psi(expected, actual)
        assert psi > 0.2

    def test_psi_interpretation(self):
        from src.drift_monitor import interpret_psi
        assert interpret_psi(0.05)[0] == "stable"
        assert interpret_psi(0.15)[0] == "warning"
        assert interpret_psi(0.25)[0] == "critical"


# ── HITL Queue ─────────────────────────────────────────────────────────────────

class TestHITLQueue:
    def test_enqueue_and_retrieve(self, tmp_path, monkeypatch):
        import src.hitl_queue as hq
        monkeypatch.setattr(hq, "QUEUE_PATH", tmp_path / "hitl_queue.csv")
        item_id = hq.enqueue("pred_001", 8.0, 30, 5, 0.55, 0.45, 0.65, "Borderline case")
        queue = hq.get_queue("pending")
        assert len(queue) == 1
        assert queue[0]["id"] == item_id

    def test_resolve_queue_item(self, tmp_path, monkeypatch):
        import src.hitl_queue as hq
        monkeypatch.setattr(hq, "QUEUE_PATH", tmp_path / "hitl_queue2.csv")
        item_id = hq.enqueue("pred_002", 5.0, 25, 2, 0.58, 0.48, 0.68, "Low CI confidence")
        result = hq.resolve(item_id, "Approve", "Applicant has collateral")
        assert result is True
        resolved = hq.get_queue("resolved")
        assert len(resolved) == 1
        assert resolved[0]["analyst_decision"] == "Approve"


# ── Validation ─────────────────────────────────────────────────────────────────

class TestValidation:
    def test_valid_inputs_no_errors(self):
        from src.validation import validate_inputs
        assert validate_inputs(0.5, 0.3, 0.2) == []

    def test_out_of_range_income(self):
        from src.validation import validate_inputs
        errors = validate_inputs(1.5, 0.3, 0.2)
        assert len(errors) > 0

    def test_experience_exceeds_age(self):
        from src.validation import validate_inputs
        errors = validate_inputs(0.5, 0.1, 0.9)
        assert len(errors) > 0

    def test_hash_is_deterministic(self):
        from src.validation import hash_input
        h1 = hash_input(0.5, 0.3, 0.2)
        h2 = hash_input(0.5, 0.3, 0.2)
        assert h1 == h2

    def test_hash_different_inputs(self):
        from src.validation import hash_input
        assert hash_input(0.5, 0.3, 0.2) != hash_input(0.6, 0.3, 0.2)
