def clean_data(df):
    df['Risk_Flag'] = df['Risk_Flag'].astype(int)

    for col in ['CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS']:
        if col in df.columns:
            df[col] = df[col].astype(float).round().astype(int)

    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])

    return df