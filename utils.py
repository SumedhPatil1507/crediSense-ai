import pandas as pd

def build_full_input(user_input, columns):

    df = pd.DataFrame([user_input])

    # Add missing columns
    for col in columns:
        if col not in df.columns:
            df[col] = 0

    # Default values
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

    # 🔥 ALL FEATURE ENGINEERING 

    if "income_per_job_year" in df.columns:
        df["income_per_job_year"] = df["Income"] / (df["CURRENT_JOB_YRS"] + 1)

    if "total_stability" in df.columns:
        df["total_stability"] = df["CURRENT_JOB_YRS"] + df["CURRENT_HOUSE_YRS"]

    if "experience_ratio" in df.columns:
        df["experience_ratio"] = df["Experience"] / (df["Age"] + 1)

    if "income_per_experience" in df.columns:
        df["income_per_experience"] = df["Income"] / (df["Experience"] + 1)

    if "stability_score" in df.columns:
        df["stability_score"] = df["CURRENT_JOB_YRS"] + df["CURRENT_HOUSE_YRS"]

    if "income_stability" in df.columns:
        df["income_stability"] = df["Income"] * df["CURRENT_JOB_YRS"]

    if "low_income_flag" in df.columns:
        df["low_income_flag"] = (df["Income"] < 0.3).astype(int)

    if "high_stability_flag" in df.columns:
        df["high_stability_flag"] = (df["CURRENT_JOB_YRS"] > 2).astype(int)

    if "Id" in df.columns:
        df["Id"] = 0

    # Final alignment
    df = df[columns]

    return df