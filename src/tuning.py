from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier

def tune_model(X, y):

    model = LGBMClassifier()

    params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1]
    }

    search = RandomizedSearchCV(
        model,
        params,
        n_iter=5,
        scoring="roc_auc",
        cv=3,
        verbose=1
    )

    search.fit(X, y)

    return search.best_estimator_