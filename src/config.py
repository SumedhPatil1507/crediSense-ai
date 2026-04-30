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
DB_PATH      = str(BASE_DIR / "data" / "credisense.db")
TARGET       = "Risk_Flag"

# API security
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "dev-secret-change-in-prod")
API_VERSION    = "v1"

# Decision thresholds
THRESHOLD_APPROVE = 0.30
THRESHOLD_REVIEW  = 0.60

# Business defaults
DEFAULT_LOAN_AMOUNT = 500_000
DEFAULT_LGD         = 0.60   # Loss Given Default

# Alerting (optional — set in environment to enable)
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")
ALERT_EMAIL       = os.getenv("ALERT_EMAIL", "")
SMTP_HOST         = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT         = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER         = os.getenv("SMTP_USER", "")
SMTP_PASS         = os.getenv("SMTP_PASS", "")
