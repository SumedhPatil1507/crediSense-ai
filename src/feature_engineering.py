def create_features(df):
    df['income_per_job_year'] = df['Income'] / (df['CURRENT_JOB_YRS'] + 1)
    df['total_stability'] = df['CURRENT_JOB_YRS'] + df['CURRENT_HOUSE_YRS']
    df['experience_ratio'] = df['Experience'] / (df['Age'] + 1)
    return df