import json
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier

TUNING_RESULTS_PATH = "models/tuning_results.json"

def tune_model(X, y, save_results=True):
    model = LGBMClassifier(class_weight="balanced")

    params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "subsample": [0.7, 0.8, 1.0],
    }

    search = RandomizedSearchCV(
        model, params, n_iter=10,
        scoring="roc_auc", cv=3,
        verbose=1, random_state=42, n_jobs=-1
    )
    search.fit(X, y)

    if save_results:
        results = pd.DataFrame(search.cv_results_)
        top = results[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
        top = top.sort_values("rank_test_score").head(10)
        top["params"] = top["params"].astype(str)
        with open(TUNING_RESULTS_PATH, "w") as f:
            json.dump(top.to_dict(orient="records"), f, indent=2)

    return search.best_estimator_, search.best_params_, search.best_score_
