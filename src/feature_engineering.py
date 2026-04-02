import numpy as np

def create_features(df):

    # Basic
    df['income_per_job_year'] = df['Income'] / (df['CURRENT_JOB_YRS'] + 1)
    df['total_stability'] = df['CURRENT_JOB_YRS'] + df['CURRENT_HOUSE_YRS']

    # 🔥 Advanced interactions
    df['income_stability'] = df['Income'] * df['total_stability']
    df['experience_ratio'] = df['Experience'] / (df['Age'] + 1)

    # 🔥 Risk indicators
    df['low_income_flag'] = (df['Income'] < 0.3).astype(int)
    df['high_stability_flag'] = (df['total_stability'] > 10).astype(int)

    return df