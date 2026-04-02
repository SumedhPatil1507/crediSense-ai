from sklearn.metrics import roc_auc_score, f1_score

def evaluate(model, X_test, y_test, loan=100000, interest=0.2):

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ML metrics
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # 🔥 Business metric
    profit = ((1 - y_prob) * loan * interest - y_prob * loan).mean()

    return {
        "F1": f1,
        "AUC": auc,
        "Expected Profit": profit
    }