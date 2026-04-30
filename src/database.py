"""
Database layer — Supabase (PostgreSQL).
Requires SUPABASE_URL and SUPABASE_KEY to be set in environment or Streamlit secrets.

Supabase setup:
  1. Create project at supabase.com (free tier)
  2. Run the SQL in db/schema.sql in the Supabase SQL editor
  3. Set SUPABASE_URL and SUPABASE_KEY in environment / Streamlit secrets
"""
import json
from datetime import datetime
from src.config import SUPABASE_URL, SUPABASE_KEY

# Try Supabase
try:
    from supabase import create_client, Client
    _sb: Client | None = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
except Exception:
    _sb = None

USING_SUPABASE = _sb is not None

def _check_supabase():
    if not USING_SUPABASE:
        raise RuntimeError("Supabase is not configured. Please set SUPABASE_URL and SUPABASE_KEY.")

# ── Public API ─────────────────────────────────────────────────────────────────

def log_prediction(income_lpa: float, age_years: int, experience_years: int,
                   income_norm: float, age_norm: float, experience_norm: float,
                   risk_prob: float, ci_lower: float, ci_upper: float,
                   decision: str, confidence: str, page: str = "API") -> str:
    """Insert a prediction record. Returns the record ID."""
    _check_supabase()
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
    _sb.table("predictions").insert(row).execute()
    return rec_id


def log_feedback(prediction_id: str, feedback: str,
                 corrected_label: str = "", notes: str = "",
                 user_id: str = "anonymous") -> None:
    _check_supabase()
    import uuid
    row = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.utcnow().isoformat(),
        "prediction_id": prediction_id,
        "feedback": feedback,
        "corrected_label": corrected_label,
        "notes": notes,
    }
    _sb.table("feedback").insert(row).execute()


def log_audit(event: str, input_hash: str, details: str = "",
              user_id: str = "anonymous") -> None:
    _check_supabase()
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
    _sb.table("audit_log").insert(row).execute()


def get_predictions(limit: int = 500) -> list[dict]:
    _check_supabase()
    res = _sb.table("predictions").select("*").order("timestamp", desc=True).limit(limit).execute()
    return res.data


def get_feedback(limit: int = 500) -> list[dict]:
    _check_supabase()
    res = _sb.table("feedback").select("*").order("timestamp", desc=True).limit(limit).execute()
    return res.data


def get_audit(limit: int = 500) -> list[dict]:
    _check_supabase()
    res = _sb.table("audit_log").select("*").order("timestamp", desc=True).limit(limit).execute()
    return res.data


def db_status() -> dict:
    return {
        "backend": "supabase" if USING_SUPABASE else "not_configured",
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_KEY),
    }
