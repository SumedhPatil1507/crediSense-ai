"""Point-in-time feature retrieval must exclude events after each entity timestamp."""
from datetime import datetime, timezone

import pandas as pd
import pytest

feast = pytest.importorskip("feast")
from feast import FeatureStore


@pytest.mark.integration
def test_historical_features_are_point_in_time(tmp_path):
    """A later correction is unavailable to an earlier application snapshot."""
    pytest.importorskip("pyarrow")
    import shutil
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    events = pd.DataFrame([
        {"applicant_id": "a-1", "pan_hash": "hash-1",
         "event_timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
         "created_timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
         "rolling_credit_utilization": 0.20, "payment_velocity": 0.90,
         "debt_to_income_ratio": 0.30},
        {"applicant_id": "a-1", "pan_hash": "hash-1",
         "event_timestamp": datetime(2024, 2, 1, tzinfo=timezone.utc),
         "created_timestamp": datetime(2024, 2, 1, tzinfo=timezone.utc),
         "rolling_credit_utilization": 0.95, "payment_velocity": 0.10,
         "debt_to_income_ratio": 0.80},
        # A late-arriving correction describes an earlier event but was not
        # available to the system at the application's historical cutoff.
        {"applicant_id": "a-1", "pan_hash": "hash-1",
         "event_timestamp": datetime(2024, 1, 10, tzinfo=timezone.utc),
         "created_timestamp": datetime(2024, 1, 20, tzinfo=timezone.utc),
         "rolling_credit_utilization": 0.88, "payment_velocity": 0.15,
         "debt_to_income_ratio": 0.75},
    ])
    events.to_parquet(data_dir / "credit_feature_events.parquet", index=False)

    repo = tmp_path / "repo"
    repo.mkdir()
    shutil.copy(root / "feature_store.yaml", repo / "feature_store.yaml")
    shutil.copy(root / "feature_repo" / "features.py", repo / "features.py")
    (repo / "feature_store.yaml").write_text(
        "project: test_pit\nprovider: local\nregistry: " +
        str(tmp_path / "registry.db").replace("\\", "/") +
        "\noffline_store:\n  type: file\n", encoding="utf-8")
    features_module = (repo / "features.py").read_text(encoding="utf-8")
    features_module = features_module.replace(
        'path="data/credit_feature_events.parquet"',
        f'path="{(data_dir / "credit_feature_events.parquet").as_posix()}"')
    (repo / "features.py").write_text(features_module, encoding="utf-8")

    store = FeatureStore(repo_path=str(repo))
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("pit_test_features", repo / "features.py")
    feature_defs = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = feature_defs
    spec.loader.exec_module(feature_defs)
    store.apply([
        feature_defs.applicant, feature_defs.pan,
        feature_defs.rolling_credit_features,
        feature_defs.payment_velocity_features,
        feature_defs.debt_to_income_features,
    ])
    entity_rows = pd.DataFrame([{
        "applicant_id": "a-1", "pan_hash": "hash-1",
        "event_timestamp": datetime(2024, 1, 15, tzinfo=timezone.utc),
    }])
    result = store.get_historical_features(
        entity_df=entity_rows,
        features=[
            "rolling_credit_features:rolling_credit_utilization",
            "payment_velocity_features:payment_velocity",
            "debt_to_income_features:debt_to_income_ratio",
        ],
        filter_by_created_timestamp=True,
    ).to_df()

    row = result.iloc[0]
    assert row["rolling_credit_utilization"] == pytest.approx(0.20)
    assert row["payment_velocity"] == pytest.approx(0.90)
    assert row["debt_to_income_ratio"] == pytest.approx(0.30)
