import pandas as pd

def build_full_input(user_input, columns):

    df = pd.DataFrame([user_input])

    # Add missing columns with defaults
    for col in columns:
        if col not in df.columns:
            df[col] = 0

    # Default values for categorical/contextual fields
    defaults = {
        "CURRENT_JOB_YRS": 2,
        "CURRENT_HOUSE_YRS": 3,
        "House_Ownership": "owned",
        "Married/Single": "single",
        "Car_Ownership": "no",
        "Profession": "Engineer",
        "CITY": "Mumbai",
        "STATE": "Maharashtra",
        "age_group": "Middle"
    }

    for col, val in defaults.items():
        if col in df.columns:
            df[col] = val

    # Feature engineering — always compute, then align to expected columns
    df["income_per_job_year"] = df["Income"] / (df["CURRENT_JOB_YRS"] + 1)
    df["total_stability"] = df["CURRENT_JOB_YRS"] + df["CURRENT_HOUSE_YRS"]
    df["experience_ratio"] = df["Experience"] / (df["Age"] + 1)
    df["income_per_experience"] = df["Income"] / (df["Experience"] + 1)
    df["stability_score"] = df["CURRENT_JOB_YRS"] + df["CURRENT_HOUSE_YRS"]
    df["income_stability"] = df["Income"] * df["CURRENT_JOB_YRS"]
    # Use a fixed low-income threshold (30% of a typical normalized income range)
    df["low_income_flag"] = (df["Income"] < 0.3).astype(int)
    df["high_stability_flag"] = (df["CURRENT_JOB_YRS"] > 2).astype(int)

    if "Id" in df.columns:
        df["Id"] = 0

    # Final alignment to expected columns
    df = df[columns]

    return df
