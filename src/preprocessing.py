import numpy as np

def clean_data(df):
    # Target
    df['Risk_Flag'] = df['Risk_Flag'].astype(int)

    # Fix numeric columns
    for col in ['CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS']:
        if col in df.columns:
            df[col] = df[col].astype(float).round().astype(int)

    # Handle extreme values (robust)
    df['Income'] = df['Income'].clip(0, 1)
    df['Age'] = df['Age'].clip(0, 1)

    return df