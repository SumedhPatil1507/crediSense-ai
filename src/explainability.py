import shap
import numpy as np


def get_explainer_and_values(pipeline, X_df):
    """
    Returns (explainer, shap_values_class1, X_dense, feature_names).
    X_df must be aligned to preprocessor's expected columns.
    """
    pre = pipeline.named_steps['preprocessor']
    mod = pipeline.named_steps['model']

    X_transformed = pre.transform(X_df)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = pre.get_feature_names_out().tolist()

    explainer = shap.TreeExplainer(mod)
    shap_values = explainer.shap_values(X_transformed)

    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    return explainer, sv, X_transformed, feature_names


def get_shap(pipeline, X_df):
    """Legacy helper — returns shap values for class 1."""
    _, sv, _, _ = get_explainer_and_values(pipeline, X_df)
    return sv
