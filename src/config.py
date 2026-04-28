"""
Centralised configuration using environment variables.
All secrets come from environment — never hardcoded.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

# Paths
DATA_PATH    = str(BASE_DIR / "data" / "loan_cleaned.csv")
MODEL_PATH   = str(BASE_DIR / "models" / "model.pkl")
COLUMNS_PATH = str(BASE_DIR / "models" / "columns.json")
TARGET       = "Risk_Flag"

# Supabase (set in environment or .env)
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# API
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "dev-secret-change-in-prod")
API_VERSION    = "v1"

# Decision thresholds
THRESHOLD_APPROVE = 0.30
THRESHOLD_REVIEW  = 0.60

# Business defaults
DEFAULT_LOAN_AMOUNT = 500_000
DEFAULT_LGD         = 0.60   # Loss Given Default
