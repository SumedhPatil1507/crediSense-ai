"""
Shadow Mode Deployment.
Runs a challenger model alongside the champion silently.
Logs both predictions for comparison without affecting live decisions.
"""
import csv
import uuid
import warnings
from pathlib import Path
from datetime import datetime

BASE_DIR    = Path(__file__).resolve().parents[1]
SHADOW_PATH = BASE_DIR / "data" / "shadow_log.csv"
HEADERS     = ["id", "timestamp", "income_norm", "age_norm", "experience_norm",
               "champion_prob", "champion_decision",
               "challenger_prob", "challenger_decision", "agreement"]


def _ensure():
    if not SHADOW_PATH.exists():
        SHADOW_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SHADOW_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=HEADERS).writeheader()


def run_shadow(champion_model, challenger_model, df_input,
               income_n: float, age_n: float, exp_n: float,
               threshold_approve: float = 0.30,
               threshold_review: float = 0.60) -> dict:
    """
    Run both models on the same input.
    Returns champion result (used for live decision) + shadow comparison.
    """
    def decide(prob):
        if prob < threshold_approve: return "Approve"
        if prob < threshold_review:  return "Manual Review"
        return "Reject"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        champ_prob = float(champion_model.predict_proba(df_input)[0][1])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chal_prob = float(challenger_model.predict_proba(df_input)[0][1])
    except Exception:
        chal_prob = None

    champ_dec = decide(champ_prob)
    chal_dec  = decide(chal_prob) if chal_prob is not None else "N/A"
    agreement = champ_dec == chal_dec if chal_prob is not None else None

    _ensure()
    row = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.utcnow().isoformat(),
        "income_norm": round(income_n, 4),
        "age_norm": round(age_n, 4),
        "experience_norm": round(exp_n, 4),
        "champion_prob": round(champ_prob, 4),
        "champion_decision": champ_dec,
        "challenger_prob": round(chal_prob, 4) if chal_prob is not None else "",
        "challenger_decision": chal_dec,
        "agreement": str(agreement),
    }
    with open(SHADOW_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=HEADERS).writerow(row)

    return {
        "champion_prob": champ_prob,
        "champion_decision": champ_dec,
        "challenger_prob": chal_prob,
        "challenger_decision": chal_dec,
        "agreement": agreement,
    }


def get_shadow_log() -> list[dict]:
    _ensure()
    with open(SHADOW_PATH) as f:
        return list(csv.DictReader(f))


def shadow_agreement_rate() -> float | None:
    rows = get_shadow_log()
    if not rows:
        return None
    agreements = [r for r in rows if r["agreement"] == "True"]
    return round(len(agreements) / len(rows), 3)
