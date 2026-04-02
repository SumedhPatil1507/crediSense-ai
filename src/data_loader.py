import pandas as pd

def load_data(path):
    df = pd.read_csv(path, sep=None, engine='python', on_bad_lines='skip')
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    print(f"✅ Data Loaded: {df.shape}")
    return df