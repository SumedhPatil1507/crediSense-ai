import joblib
from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.preprocessing import clean_data
from src.feature_engineering import create_features
from src.encoding import build_preprocessor
from src.pipeline import build_pipeline
from src.config import *

def train():

    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = create_features(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = build_preprocessor(X)
    pipeline = build_pipeline(preprocessor)

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, MODEL_PATH)

    print("✅ Model trained & saved")

if __name__ == "__main__":
    train()