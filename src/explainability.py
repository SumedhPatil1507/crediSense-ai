import shap

def get_shap(pipeline, X):
    model = pipeline.named_steps['model']
    pre = pipeline.named_steps['preprocessor']

    X_transformed = pre.transform(X)

    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X_transformed)