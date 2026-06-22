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

# Alerting
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")
ALERT_EMAIL       = os.getenv("ALERT_EMAIL", "")
SMTP_HOST         = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT         = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER         = os.getenv("SMTP_USER", "")
SMTP_PASS         = os.getenv("SMTP_PASS", "")

# AES-256 Encryption (set ENCRYPTION_KEY to enable field-level encryption)
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "")

# S3-compatible Model Registry (AWS S3 / Azure Blob / MinIO)
# Leave blank to use local registry.json fallback
S3_BUCKET      = os.getenv("S3_BUCKET", "")
S3_PREFIX      = os.getenv("S3_PREFIX", "credisense/models/")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION     = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
S3_ENDPOINT    = os.getenv("S3_ENDPOINT_URL", "")  # for MinIO/Azure

# Redis / Celery (for async task queue — optional)
REDIS_URL = os.getenv("REDIS_URL", "")

# Credit Bureau API (CIBIL / Experian / Equifax)
BUREAU_API_KEY  = os.getenv("CREDIT_BUREAU_API_KEY", "")
BUREAU_API_URL  = os.getenv("CREDIT_BUREAU_URL", "https://api.cibil.com/v1")
BUREAU_MOCK     = os.getenv("BUREAU_MOCK", "true").lower() == "true"

# Prometheus metrics
PROMETHEUS_ENABLED = os.getenv("PROMETHEUS_ENABLED", "false").lower() == "true"
