from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

def build_pipeline(preprocessor):
    model = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        class_weight='balanced'
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])