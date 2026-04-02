from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def build_preprocessor(X):
    cat_cols = X.select_dtypes(include='object').columns
    num_cols = X.select_dtypes(exclude='object').columns

    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ("num", "passthrough", num_cols)
    ])