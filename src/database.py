"""
Database layer — Supabase (PostgreSQL) with CSV fallback.
If SUPABASE_URL/KEY are set, uses Supabase. Otherwise falls back to CSV files.

Supabase setup:
  1. Create project at supabase.com (free tier)
  2. Run the SQL in db/schema.sql in the Supabase SQL editor
  3. Set SUPABASE_URL and SUPABASE_KEY in environment / Streamlit secrets
"""
import csv
import json
from pathlib import Path
from datetime import datetime
from src.config import SUPABASE_URL, SUPABASE_KEY, BASE_DIR

# Try Supabase
try:
    from supabase import create_client, Client
    _sb: Client | None = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
except Exception:
    _sb = None

USING_SUPABASE = _sb is not None

# CSV fallback paths
_CSV_PREDICTIONS = BASE_DIR / "data" / "predictions.csv"
_CSV_FEEDBACK    = BASE_DIR / "data" / "feedback_log.csv"
_CSV_AUDIT       = BASE_DIR / "data" / "audit_log.csv"

_PRED_HEADERS    = ["id","timestamp","income_lpa","age_years","experience_years",
                    "income_norm","age_norm","experience_norm",
                    "risk_prob","ci_lower","ci_upper","decision","confidence","page"]
_FB_HEADERS      = ["id","timestamp","prediction_id","feedback","corrected_label","notes"]
_AUDIT_HEADERS   = ["id","timestamp","event","input_hash","user_id","details"]


def _ensure(path: Path, headers: list):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=headers).writeheader()


def _csv_append(path: Path, headers: list, row: dict):
    _ensure(path, headers)
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=headers).writerow(row)


def _csv_read(path: Path, headers: list) -> list[dict]:
    _ensure(path, headers)
    with open(path, "r") as f:
        return list(csv.DictReader(f))


# ── Public API ─────────────────────────────────────────────────────────────────

def log_prediction(income_lpa: float, age_years: int, experience_years: int,
                   income_norm: float, age_norm: float, experience_norm: float,
                   risk_prob: float, ci_lower: float, ci_upper: float,
                   decision: str, confidence: str, page: str = "API") -> str:
    """Insert a prediction record. Returns the record ID."""
    import uuid
    rec_id = str(uuid.uuid4())[:8]
    row = {
        "id": rec_id,
        "timestamp": datetime.utcnow().isoformat(),
        "income_lpa": round(income_lpa, 2),
        "age_years": age_years,
        "experience_years": experience_years,
        "income_norm": round(income_norm, 4),
        "age_norm": round(age_norm, 4),
        "experience_norm": round(experience_norm, 4),
        "risk_prob": round(risk_prob, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "decision": decision,
        "confidence": confidence,
        "page": page,
    }
    if USING_SUPABASE:
        try:
            _sb.table("predictions").insert(row).execute()
            return rec_id
        except Exception:
            pass
    _csv_append(_CSV_PREDICTIONS, _PRED_HEADERS, row)
    return rec_id


def log_feedback(prediction_id: str, feedback: str,
                 corrected_label: str = "", notes: str = "",
                 user_id: str = "anonymous") -> None:
    import uuid
    row = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.utcnow().isoformat(),
        "prediction_id": prediction_id,
        "feedback": feedback,
        "corrected_label": corrected_label,
        "notes": notes,
    }
    if USING_SUPABASE:
        try:
            _sb.table("feedback").insert(row).execute()
            return
        except Exception:
            pass
    _csv_append(_CSV_FEEDBACK, _FB_HEADERS, row)


def log_audit(event: str, input_hash: str, details: str = "",
              user_id: str = "anonymous") -> None:
    import uuid
    import hashlib
    row = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.utcnow().isoformat(),
        "event": event,
        "input_hash": input_hash,
        "user_id": hashlib.sha256(user_id.encode()).hexdigest()[:12],
        "details": details,
    }
    if USING_SUPABASE:
        try:
            _sb.table("audit_log").insert(row).execute()
            return
        except Exception:
            pass
    _csv_append(_CSV_AUDIT, _AUDIT_HEADERS, row)


def get_predictions(limit: int = 500) -> list[dict]:
    if USING_SUPABASE:
        try:
            res = _sb.table("predictions").select("*").order("timestamp", desc=True).limit(limit).execute()
            return res.data
        except Exception:
            pass
    return _csv_read(_CSV_PREDICTIONS, _PRED_HEADERS)[-limit:]


def get_feedback(limit: int = 500) -> list[dict]:
    if USING_SUPABASE:
        try:
            res = _sb.table("feedback").select("*").order("timestamp", desc=True).limit(limit).execute()
            return res.data
        except Exception:
            pass
    return _csv_read(_CSV_FEEDBACK, _FB_HEADERS)[-limit:]


def get_audit(limit: int = 500) -> list[dict]:
    if USING_SUPABASE:
        try:
            res = _sb.table("audit_log").select("*").order("timestamp", desc=True).limit(limit).execute()
            return res.data
        except Exception:
            pass
    return _csv_read(_CSV_AUDIT, _AUDIT_HEADERS)[-limit:]


def db_status() -> dict:
    return {
        "backend": "supabase" if USING_SUPABASE else "csv_fallback",
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_KEY),
    }
