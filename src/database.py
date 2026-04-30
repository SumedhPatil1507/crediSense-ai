"""
Database layer — Local SQLite (Premium Tier Zero-Config DB).
Automatically creates and manages a robust local database for persistence.
"""
import sqlite3
import json
import logging
import hashlib
import uuid
from datetime import datetime
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "credisense.db"

def _ensure_db():
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                income_lpa REAL,
                age_years INTEGER,
                experience_years INTEGER,
                income_norm REAL,
                age_norm REAL,
                experience_norm REAL,
                risk_prob REAL,
                ci_lower REAL,
                ci_upper REAL,
                decision TEXT,
                confidence TEXT,
                page TEXT
            )
        """)
        # Feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                prediction_id TEXT,
                feedback TEXT,
                corrected_label TEXT,
                notes TEXT
            )
        """)
        # Audit log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                event TEXT,
                input_hash TEXT,
                user_id TEXT,
                details TEXT
            )
        """)
        conn.commit()

# Ensure DB is created on load
_ensure_db()

def _dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

# ── Public API ─────────────────────────────────────────────────────────────────

def log_prediction(income_lpa: float, age_years: int, experience_years: int,
                   income_norm: float, age_norm: float, experience_norm: float,
                   risk_prob: float, ci_lower: float, ci_upper: float,
                   decision: str, confidence: str, page: str = "API") -> str:
    """Insert a prediction record. Returns the record ID."""
    rec_id = str(uuid.uuid4())[:8]
    row = (
        rec_id,
        datetime.utcnow().isoformat(),
        round(income_lpa, 2),
        age_years,
        experience_years,
        round(income_norm, 4),
        round(age_norm, 4),
        round(experience_norm, 4),
        round(risk_prob, 4),
        round(ci_lower, 4),
        round(ci_upper, 4),
        decision,
        confidence,
        page
    )
    
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, row)
        conn.commit()
    return rec_id


def log_feedback(prediction_id: str, feedback: str,
                 corrected_label: str = "", notes: str = "",
                 user_id: str = "anonymous") -> None:
    row = (
        str(uuid.uuid4())[:8],
        datetime.utcnow().isoformat(),
        prediction_id,
        feedback,
        corrected_label,
        notes
    )
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?)
        """, row)
        conn.commit()


def log_audit(event: str, input_hash: str, details: str = "",
              user_id: str = "anonymous") -> None:
    row = (
        str(uuid.uuid4())[:8],
        datetime.utcnow().isoformat(),
        event,
        input_hash,
        hashlib.sha256(user_id.encode()).hexdigest()[:12],
        details
    )
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO audit_log VALUES (?, ?, ?, ?, ?, ?)
        """, row)
        conn.commit()


def get_predictions(limit: int = 500) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = _dict_factory
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?", (limit,))
        return cursor.fetchall()


def get_feedback(limit: int = 500) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = _dict_factory
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM feedback ORDER BY timestamp DESC LIMIT ?", (limit,))
        return cursor.fetchall()


def get_audit(limit: int = 500) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = _dict_factory
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?", (limit,))
        return cursor.fetchall()


def db_status() -> dict:
    return {
        "backend": "sqlite",
        "db_path": str(DB_PATH),
        "db_size_bytes": DB_PATH.stat().st_size if DB_PATH.exists() else 0
    }
