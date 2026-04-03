def create_features(df):
    df['income_per_job_year'] = df['Income'] / (df['CURRENT_JOB_YRS'] + 1)
    df['total_stability'] = df['CURRENT_JOB_YRS'] + df['CURRENT_HOUSE_YRS']
    df['experience_ratio'] = df['Experience'] / (df['Age'] + 1)
    df['income_per_experience'] = df['Income'] / (df['Experience'] + 1)
    df['stability_score'] = df['CURRENT_JOB_YRS'] + df['CURRENT_HOUSE_YRS']
    df['income_stability'] = df['Income'] * df['CURRENT_JOB_YRS']
    df['low_income_flag'] = (df['Income'] < df['Income'].quantile(0.3)).astype(int)
    df['high_stability_flag'] = (df['CURRENT_JOB_YRS'] > 2).astype(int)
    return df
