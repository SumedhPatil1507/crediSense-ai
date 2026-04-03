import shap

def get_shap(pipeline, X):
    model = pipeline.named_steps['model']
    pre = pipeline.named_steps['preprocessor']

    X_transformed = pre.transform(X)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    # For binary classification, shap_values is a list of 2 arrays; return class 1
    if isinstance(shap_values, list):
        return shap_values[1]
    return shap_values
