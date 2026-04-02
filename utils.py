import pandas as pd

def build_full_input(user_input, columns):

    df = pd.DataFrame(columns=columns)
    df.loc[0] = 0

    # Fill user inputs
    for key, value in user_input.items():
        if key in df.columns:
            df[key] = value

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

    # Feature engineering
    if "income_per_job_year" in df.columns:
        df["income_per_job_year"] = df["Income"] / (df["CURRENT_JOB_YRS"] + 1)

    if "total_stability" in df.columns:
        df["total_stability"] = df["CURRENT_JOB_YRS"] + df["CURRENT_HOUSE_YRS"]

    if "experience_ratio" in df.columns:
        df["experience_ratio"] = df["Experience"] / (df["Age"] + 1)

    return df