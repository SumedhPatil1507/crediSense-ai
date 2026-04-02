import shap

def get_shap(pipeline, X_sample):

    model = pipeline.named_steps['model']
    pre = pipeline.named_steps['preprocessor']

    X_transformed = pre.transform(X_sample)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    return shap_values