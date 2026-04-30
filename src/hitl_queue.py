"""
Human-in-the-Loop (HITL) Queue.
Cases flagged for manual review are queued here.
Analysts can approve/reject/escalate from the UI.
"""
import csv
import uuid
from pathlib import Path
from datetime import datetime

BASE_DIR  = Path(__file__).resolve().parents[1]
QUEUE_PATH = BASE_DIR / "data" / "hitl_queue.csv"
HEADERS   = ["id", "timestamp", "prediction_id", "income_lpa", "age_years",
             "experience_years", "risk_prob", "ci_lower", "ci_upper",
             "reason", "status", "analyst_decision", "analyst_notes", "resolved_at"]


def _ensure():
    if not QUEUE_PATH.exists():
        QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(QUEUE_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=HEADERS).writeheader()


def enqueue(prediction_id: str, income_lpa: float, age_years: int,
            experience_years: int, risk_prob: float,
            ci_lower: float, ci_upper: float, reason: str) -> str:
    """Add a case to the HITL queue. Returns queue item ID."""
    _ensure()
    item_id = str(uuid.uuid4())[:8]
    row = {
        "id": item_id,
        "timestamp": datetime.utcnow().isoformat(),
        "prediction_id": prediction_id,
        "income_lpa": round(income_lpa, 2),
        "age_years": age_years,
        "experience_years": experience_years,
        "risk_prob": round(risk_prob, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "reason": reason,
        "status": "pending",
        "analyst_decision": "",
        "analyst_notes": "",
        "resolved_at": "",
    }
    with open(QUEUE_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=HEADERS).writerow(row)
    return item_id


def get_queue(status: str = "pending") -> list[dict]:
    _ensure()
    with open(QUEUE_PATH) as f:
        rows = list(csv.DictReader(f))
    if status == "all":
        return rows
    return [r for r in rows if r["status"] == status]


def resolve(item_id: str, decision: str, notes: str = "") -> bool:
    """Resolve a queue item with analyst decision."""
    _ensure()
    rows = []
    found = False
    with open(QUEUE_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["id"] == item_id:
                row["status"] = "resolved"
                row["analyst_decision"] = decision
                row["analyst_notes"] = notes
                row["resolved_at"] = datetime.utcnow().isoformat()
                found = True
            rows.append(row)

    if found:
        with open(QUEUE_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=HEADERS)
            writer.writeheader()
            writer.writerows(rows)
    return found


def queue_stats() -> dict:
    all_items = get_queue(status="all")
    return {
        "total": len(all_items),
        "pending": sum(1 for r in all_items if r["status"] == "pending"),
        "resolved": sum(1 for r in all_items if r["status"] == "resolved"),
    }

def agentic_review(item_data: dict) -> dict:
    """
    Simulates an AI Agent providing a recommended decision and rationale
    based on the applicant's data and risk probability.
    """
    prob = float(item_data.get("risk_prob", 0.5))
    income = float(item_data.get("income_lpa", 0))
    exp = int(item_data.get("experience_years", 0))
    
    # Simple rule-based expert system simulating an LLM analysis
    if prob > 0.65:
        decision = "Reject"
        rationale = f"Risk probability is high ({prob:.1%}). "
    elif prob < 0.35:
        decision = "Approve"
        rationale = f"Risk probability is low ({prob:.1%}). "
    else:
        decision = "Manual Review"
        rationale = f"Risk probability is borderline ({prob:.1%}). "
        
    if income > 15:
        rationale += "Applicant has strong income, which may offset some risk. "
    elif income < 5:
        rationale += "Income is relatively low, increasing exposure. "
        
    if exp > 10:
        rationale += "Extensive work experience suggests stability."
    elif exp < 2:
        rationale += "Limited work experience poses additional risk."
        
    return {
        "recommended_decision": decision,
        "rationale": rationale
    }
