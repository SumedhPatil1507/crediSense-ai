"""Input validation, PII masking, and audit logging."""
import re
import csv
import hashlib
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[1]
AUDIT_PATH = BASE_DIR / "data" / "audit_log.csv"
AUDIT_HEADERS = ["timestamp", "event", "user_hash", "input_hash", "details"]


def validate_inputs(income: float, age: float, experience: float) -> list[str]:
    """Returns list of validation error messages. Empty = valid."""
    errors = []
    if not (0.0 <= income <= 1.0):
        errors.append(f"Income must be between 0 and 1 (got {income})")
    if not (0.0 <= age <= 1.0):
        errors.append(f"Age must be between 0 and 1 (got {age})")
    if not (0.0 <= experience <= 1.0):
        errors.append(f"Experience must be between 0 and 1 (got {experience})")
    if experience > age:
        errors.append("Experience cannot exceed Age (normalized values)")
    return errors


def mask_pii(value: float, label: str) -> str:
    """Mask sensitive numeric values for display in logs."""
    if label == "income":
        return f"[INCOME: {'High' if value > 0.6 else 'Mid' if value > 0.3 else 'Low'}]"
    if label == "age":
        return f"[AGE: {'Senior' if value > 0.7 else 'Mid' if value > 0.35 else 'Young'}]"
    return f"[{label.upper()}: masked]"


def hash_input(income: float, age: float, experience: float) -> str:
    """Create a non-reversible hash of inputs for audit trail."""
    raw = f"{income:.4f}|{age:.4f}|{experience:.4f}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def audit_log(event: str, income: float, age: float, experience: float,
              details: str = "", user_id: str = "anonymous"):
    """Write an audit entry."""
    if not AUDIT_PATH.exists():
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=AUDIT_HEADERS).writeheader()
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        "user_hash": hashlib.sha256(user_id.encode()).hexdigest()[:12],
        "input_hash": hash_input(income, age, experience),
        "details": details,
    }
    with open(AUDIT_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=AUDIT_HEADERS).writerow(row)


def load_audit() -> list[dict]:
    if not AUDIT_PATH.exists():
        return []
    with open(AUDIT_PATH, "r") as f:
        return list(csv.DictReader(f))
