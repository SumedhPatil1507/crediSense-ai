"""
Feedback loop + usage logging.
Stores predictions and user feedback in local CSV files.
On Streamlit Cloud these persist within a session but reset on redeploy —
for true persistence, swap the CSV writes for a database call.
"""
import csv
import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[1]
FEEDBACK_PATH = BASE_DIR / "data" / "feedback_log.csv"
USAGE_PATH = BASE_DIR / "data" / "usage_log.csv"

FEEDBACK_HEADERS = ["timestamp", "income", "age", "experience", "risk_prob",
                    "decision", "feedback", "corrected_label", "notes"]
USAGE_HEADERS = ["timestamp", "page", "income", "age", "experience",
                 "risk_prob", "decision", "confidence_band"]


def _ensure_file(path: Path, headers: list):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=headers).writeheader()


def log_prediction(income: float, age: float, experience: float,
                   prob: float, decision: str, page: str = "Model"):
    """Log every prediction to usage_log.csv."""
    _ensure_file(USAGE_PATH, USAGE_HEADERS)
    band = "Low (<30%)" if prob < 0.3 else "Medium (30-60%)" if prob < 0.6 else "High (>60%)"
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "page": page,
        "income": round(income, 4),
        "age": round(age, 4),
        "experience": round(experience, 4),
        "risk_prob": round(prob, 4),
        "decision": decision,
        "confidence_band": band,
    }
    with open(USAGE_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=USAGE_HEADERS).writerow(row)


def log_feedback(income: float, age: float, experience: float,
                 prob: float, decision: str,
                 feedback: str, corrected_label: str = "", notes: str = ""):
    """Log analyst feedback on a prediction."""
    _ensure_file(FEEDBACK_PATH, FEEDBACK_HEADERS)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "income": round(income, 4),
        "age": round(age, 4),
        "experience": round(experience, 4),
        "risk_prob": round(prob, 4),
        "decision": decision,
        "feedback": feedback,
        "corrected_label": corrected_label,
        "notes": notes,
    }
    with open(FEEDBACK_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FEEDBACK_HEADERS).writerow(row)


def load_feedback() -> list[dict]:
    _ensure_file(FEEDBACK_PATH, FEEDBACK_HEADERS)
    with open(FEEDBACK_PATH, "r") as f:
        return list(csv.DictReader(f))


def load_usage() -> list[dict]:
    _ensure_file(USAGE_PATH, USAGE_HEADERS)
    with open(USAGE_PATH, "r") as f:
        return list(csv.DictReader(f))
