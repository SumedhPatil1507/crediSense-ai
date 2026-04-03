import pandas as pd

def build_full_input(user_input, columns):

    # Step 1: base input
    df = pd.DataFrame([user_input])

    # Step 2: add missing columns
    for col in columns:
        if col not in df.columns:
            df[col] = 0

    # Step 3: defaults
    df["CURRENT_JOB_YRS"] = 2
    df["CURRENT_HOUSE_YRS"] = 3
    df["House_Ownership"] = "owned"
    df["Married/Single"] = "single"
    df["Car_Ownership"] = "no"
    df["Profession"] = "Engineer"
    df["CITY"] = "Mumbai"
    df["STATE"] = "Maharashtra"
    df["age_group"] = "Middle"

    # Step 4: feature engineering (MATCH TRAINING)

    # income_per_experience
    if "income_per_experience" in df.columns:
        df["income_per_experience"] = df["Income"] / (df["Experience"] + 1)

    # stability_score
    if "stability_score" in df.columns:
        df["stability_score"] = df["CURRENT_JOB_YRS"] + df["CURRENT_HOUSE_YRS"]

    # income_stability
    if "income_stability" in df.columns:
        df["income_stability"] = df["Income"] * df["stability_score"]

    # flags
    if "low_income_flag" in df.columns:
        df["low_income_flag"] = (df["Income"] < 0.3).astype(int)

    if "high_stability_flag" in df.columns:
        df["high_stability_flag"] = (df["stability_score"] > 5).astype(int)

    # optional Id
    if "Id" in df.columns:
        df["Id"] = 0

    # Step 5: reorder
    df = df[columns]

    return df