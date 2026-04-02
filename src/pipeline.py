from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

def build_pipeline(preprocessor):

    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        class_weight='balanced',
        subsample=0.8,
        colsample_bytree=0.8
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline