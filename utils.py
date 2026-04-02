import pandas as pd

def build_full_input(user_input, columns):

    # Step 1: Create full dataframe
    df = pd.DataFrame([user_input])

    # Step 2: Add missing columns
    for col in columns:
        if col not in df.columns:
            df[col] = 0

    # Step 3: Ensure correct column order
    df = df[columns]

    # Step 4: Default categorical values
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

    # Step 5: Feature engineering (MUST match training)
    if "income_per_job_year" in df.columns:
        df["income_per_job_year"] = df["Income"] / (df["CURRENT_JOB_YRS"] + 1)

    if "total_stability" in df.columns:
        df["total_stability"] = df["CURRENT_JOB_YRS"] + df["CURRENT_HOUSE_YRS"]

    if "experience_ratio" in df.columns:
        df["experience_ratio"] = df["Experience"] / (df["Age"] + 1)

    return df