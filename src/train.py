import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.preprocessing import clean_data
from src.feature_engineering import create_features
from src.encoding import build_preprocessor
from src.pipeline import build_pipeline
from src.config import *
from src.evaluation_report import build_holdout_report

def train():
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = create_features(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Save column names
    with open(COLUMNS_PATH, "w") as f:
        json.dump(X.columns.tolist(), f)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = build_preprocessor(X)
    pipeline = build_pipeline(preprocessor)

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, MODEL_PATH)

    report = build_holdout_report(
        pipeline, X_train, y_train, X_test, y_test,
        dataset_path=DATA_PATH, model_path=MODEL_PATH, split_seed=42,
    )
    report_path = Path(BASE_DIR) / "reports" / "generated" / "model_evaluation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("✅ Model trained & saved")

if __name__ == "__main__":
    train()
