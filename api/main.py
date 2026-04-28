"""
CrediSense AI — FastAPI Backend
Endpoints: /predict, /predict/batch, /explain, /feedback, /health, /metrics
"""
import json
import warnings
import hashlib
from pathlib import Path
from typing import Optional
import joblib

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

import sys
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src.config import (MODEL_PATH, COLUMNS_PATH, API_SECRET_KEY, API_VERSION,
                         THRESHOLD_APPROVE, THRESHOLD_REVIEW)
from src.confidence_intervals import bootstrap_ci, interpret_ci
from src.adverse_action import generate_adverse_action
from src.validation import hash_input
from src.database import log_prediction, log_feedback, log_audit, db_status
from src.hitl_queue import enqueue as hitl_enqueue
from src.model_registry import get_active_version, get_current_model_hash

# ── Load model ─────────────────────────────────────────────────────────────────
model = joblib.load(MODEL_PATH)
with open(COLUMNS_PATH) as f:
    COLS = json.load(f)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CrediSense AI",
    description="Production-grade Credit Risk Scoring API",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth ───────────────────────────────────────────────────────────────────────
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

# ── Schemas ────────────────────────────────────────────────────────────────────
class ApplicantInput(BaseModel):
    income_lpa: float = Field(..., gt=0, le=500, description="Annual income in LPA (lakhs per annum)")
    age_years: int    = Field(..., ge=18, le=70, description="Age in years")
    experience_years: int = Field(..., ge=0, le=45, description="Work experience in years")
    profession: Optional[str] = "Engineer"
    city: Optional[str] = "Mumbai"
    state: Optional[str] = "Maharashtra"
    house_ownership: Optional[str] = "owned"
    marital_status: Optional[str] = "single"
    car_ownership: Optional[str] = "no"
    current_job_years: Optional[int] = Field(2, ge=0, le=40)
    current_house_years: Optional[int] = Field(3, ge=0, le=40)

    @field_validator("experience_years")
    @classmethod
    def exp_lt_age(cls, v, info):
        age = info.data.get("age_years", 70)
        if v >= age - 16:
            raise ValueError("Experience cannot exceed working age")
        return v


class FeedbackInput(BaseModel):
    prediction_id: str
    feedback: str = Field(..., pattern="^(correct|incorrect|unsure)$")
    corrected_label: Optional[str] = ""
    notes: Optional[str] = ""


class BatchInput(BaseModel):
    applicants: list[ApplicantInput] = Field(..., max_length=500)


# ── Helpers ────────────────────────────────────────────────────────────────────
def normalize(income_lpa, age_years, experience_years):
    income_norm = min(income_lpa / 50.0, 1.0)
    age_norm    = (age_years - 18) / 52.0
    exp_norm    = min(experience_years / 40.0, 1.0)
    return income_norm, age_norm, exp_norm


def build_input_df(inp: ApplicantInput):
    from utils import build_full_input
    income_norm, age_norm, exp_norm = normalize(
        inp.income_lpa, inp.age_years, inp.experience_years)
    user_input = {
        "Income": income_norm, "Age": age_norm, "Experience": exp_norm,
        "CURRENT_JOB_YRS": inp.current_job_years,
        "CURRENT_HOUSE_YRS": inp.current_house_years,
        "House_Ownership": inp.house_ownership,
        "Married/Single": inp.marital_status,
        "Car_Ownership": inp.car_ownership,
        "Profession": inp.profession,
        "CITY": inp.city,
        "STATE": inp.state,
        "age_group": "Young" if age_norm < 0.3 else "Senior" if age_norm > 0.7 else "Middle",
    }
    return build_full_input(user_input, COLS), income_norm, age_norm, exp_norm


def make_decision(prob: float) -> str:
    if prob < THRESHOLD_APPROVE:
        return "Approve"
    elif prob < THRESHOLD_REVIEW:
        return "Manual Review"
    return "Reject"


def make_confidence(prob: float) -> str:
    margin = abs(prob - 0.5)
    if margin > 0.3:  return "High"
    if margin > 0.15: return "Medium"
    return "Low"


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": API_VERSION, "db": db_status()}


@app.post(f"/api/{API_VERSION}/predict")
def predict(inp: ApplicantInput, _: str = Depends(verify_api_key)):
    try:
        df_in, inc_n, age_n, exp_n = build_input_df(inp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob = float(model.predict_proba(df_in)[0][1])

        _, ci_lower, ci_upper = bootstrap_ci(model, df_in, n_bootstrap=100)
        decision   = make_decision(prob)
        confidence = make_confidence(prob)
        adverse    = generate_adverse_action(prob, inc_n, age_n, exp_n)

        input_hash = hash_input(inc_n, age_n, exp_n)
        pred_id = log_prediction(
            inp.income_lpa, inp.age_years, inp.experience_years,
            inc_n, age_n, exp_n, prob, ci_lower, ci_upper,
            decision, confidence, page="API"
        )
        log_audit("PREDICT", input_hash, details=f"decision={decision} prob={prob:.4f}")

        # Auto-queue borderline cases for human review
        hitl_queued = False
        if decision == "Manual Review" or confidence == "Low":
            hitl_enqueue(pred_id, inp.income_lpa, inp.age_years, inp.experience_years,
                         prob, ci_lower, ci_upper,
                         reason="Borderline probability" if decision == "Manual Review" else "Low confidence CI")
            hitl_queued = True

        # Get active model version
        active_ver = get_active_version()
        model_ver  = active_ver["version_id"] if active_ver else API_VERSION
        model_hash = get_current_model_hash()

        return {
            "prediction_id": pred_id,
            "risk_probability": round(prob, 4),
            "safe_probability": round(1 - prob, 4),
            "decision": decision,
            "confidence": confidence,
            "confidence_interval_95": {"lower": round(ci_lower, 4), "upper": round(ci_upper, 4)},
            "adverse_action": adverse if adverse["required"] else None,
            "model_version": model_ver,
            "model_hash": model_hash,
            "queued_for_review": hitl_queued,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(f"/api/{API_VERSION}/predict/batch")
def predict_batch(batch: BatchInput, _: str = Depends(verify_api_key)):
    results = []
    for inp in batch.applicants:
        try:
            df_in, inc_n, age_n, exp_n = build_input_df(inp)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prob = float(model.predict_proba(df_in)[0][1])
            decision = make_decision(prob)
            pred_id = log_prediction(
                inp.income_lpa, inp.age_years, inp.experience_years,
                inc_n, age_n, exp_n, prob, 0.0, 0.0,
                decision, make_confidence(prob), page="API_BATCH"
            )
            results.append({
                "prediction_id": pred_id,
                "risk_probability": round(prob, 4),
                "decision": decision,
                "error": None,
            })
        except Exception as e:
            results.append({"prediction_id": None, "risk_probability": None,
                             "decision": "Error", "error": str(e)})
    return {"count": len(results), "results": results}


@app.post(f"/api/{API_VERSION}/explain")
def explain(inp: ApplicantInput, _: str = Depends(verify_api_key)):
    try:
        import shap
        df_in, inc_n, age_n, exp_n = build_input_df(inp)
        pre = model.named_steps["preprocessor"]
        mod = model.named_steps["model"]
        X = pre.transform(df_in)
        if hasattr(X, "toarray"):
            X = X.toarray()
        feature_names = pre.get_feature_names_out().tolist()
        explainer = shap.TreeExplainer(mod)
        sv = explainer.shap_values(X)
        shap_vals = sv[1][0] if isinstance(sv, list) else sv[0]
        top = sorted(
            [{"feature": feature_names[i], "shap_value": round(float(shap_vals[i]), 4)}
             for i in range(len(feature_names))],
            key=lambda x: abs(x["shap_value"]), reverse=True
        )[:10]
        return {"top_features": top, "base_value": round(float(explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value), 4)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(f"/api/{API_VERSION}/feedback")
def feedback(fb: FeedbackInput, _: str = Depends(verify_api_key)):
    log_feedback(fb.prediction_id, fb.feedback, fb.corrected_label, fb.notes)
    return {"status": "recorded", "prediction_id": fb.prediction_id}


@app.get(f"/api/{API_VERSION}/metrics")
def metrics(_: str = Depends(verify_api_key)):
    from src.database import get_predictions, get_feedback
    preds = get_predictions(limit=1000)
    fbs   = get_feedback(limit=1000)
    if not preds:
        return {"total_predictions": 0}
    import statistics
    probs = [float(p["risk_prob"]) for p in preds if p.get("risk_prob")]
    decisions = [p["decision"] for p in preds if p.get("decision")]
    return {
        "total_predictions": len(preds),
        "avg_risk_score": round(statistics.mean(probs), 4) if probs else None,
        "approve_rate": round(decisions.count("Approve") / len(decisions), 3) if decisions else None,
        "reject_rate": round(decisions.count("Reject") / len(decisions), 3) if decisions else None,
        "review_rate": round(decisions.count("Manual Review") / len(decisions), 3) if decisions else None,
        "total_feedback": len(fbs),
        "db_backend": db_status()["backend"],
    }
