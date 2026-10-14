"""
CrediSense AI — FastAPI Backend
Endpoints: /predict, /predict/batch, /explain, /feedback, /health, /metrics
"""
import asyncio
import json
import os
import warnings
import hashlib
import hmac
import logging
from pathlib import Path
from typing import Optional
import joblib

from fastapi import FastAPI, HTTPException, Depends, Header, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

import sys
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src.config import (MODEL_PATH, COLUMNS_PATH, API_SECRET_KEY, API_VERSION,
                         THRESHOLD_APPROVE, THRESHOLD_REVIEW, API_ALLOWED_ORIGINS)
from src.confidence_intervals import bootstrap_ci, interpret_ci
from src.conformal import calibrated_interval
from src.adverse_action import generate_adverse_action
from src.validation import hash_input
from src.database import log_prediction, log_feedback, log_audit, db_status
from src.hitl_queue import enqueue as hitl_enqueue
from src.model_registry import get_active_version, get_current_model_hash
from src.metrics_server import setup_metrics
from src.telemetry import traced_predict_proba, traced_span
from src.streaming.prediction_events import prediction_channel
from redis.asyncio import Redis
from sse_starlette.sse import EventSourceResponse

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
    allow_origins=API_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
LOGGER = logging.getLogger(__name__)
setup_metrics(app)

# ── Auth ───────────────────────────────────────────────────────────────────────
def verify_api_key(x_api_key: str = Header(...)):
    if not hmac.compare_digest(x_api_key, API_SECRET_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

# ── Schemas ────────────────────────────────────────────────────────────────────
class ApplicantInput(BaseModel):
    applicant_id: Optional[str] = Field(
        None, min_length=1, max_length=128,
        description="Stable internal graph key used to retrieve precomputed R-GCN risk attribution",
    )
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


def standardized_risk_response(verdict: str, lower: float | None = None,
                               upper: float | None = None) -> dict:
    """Build the stable four-field risk decision contract used by score routes."""
    interval = (f"; Conformal Prediction 90% confidence interval bounds [{lower:.4f}, {upper:.4f}]"
                if lower is not None and upper is not None else
                "; Conformal Prediction 90% confidence interval bounds unavailable")
    return {
        "verdict": verdict,
        "reasoning": "LightGBM SHAP feature drivers" + interval,
        "recommendation": "Dynamic Fraction Kelly position size + ATR stop-loss distance",
        "next_steps": [
            "Step 1: Submit order to Alpaca paper/live gateway",
            "Step 2: Log trade execution to audit ledger",
            "Step 3: Monitor PSI/CSI feature drift",
        ],
    }


def score_credit_risk(input_frame, *, operation: str = "predict") -> float:
    """Run and trace the model score without attaching applicant feature values."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(traced_predict_proba(model, input_frame, operation=operation)[0][1])


def get_gnn_attribution(applicant_id: str | None) -> dict | None:
    """Return precomputed graph scores when the optional GNN/Neo4j stack is enabled."""
    if not applicant_id:
        return None
    try:
        from src.gnn_risk import get_applicant_graph_risk
        return get_applicant_graph_risk(applicant_id)
    except ImportError:
        LOGGER.info("PyTorch Geometric / Neo4j GNN stack is not installed")
        return None


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": API_VERSION, "db": db_status()}


@app.get(f"/api/{API_VERSION}/stream/predictions/{{symbol}}")
async def stream_predictions(symbol: str, request: Request,
                             _: str = Depends(verify_api_key)):
    """Stream prediction events from Redis Pub/Sub as server-sent events."""
    try:
        channel = prediction_channel(symbol)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    redis_client = Redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
    pubsub = redis_client.pubsub()
    try:
        await pubsub.subscribe(channel)
    except Exception as exc:
        await pubsub.aclose()
        await redis_client.aclose()
        LOGGER.exception("Redis prediction subscription failed")
        raise HTTPException(status_code=503, detail="Prediction stream is unavailable") from exc

    async def events():
        try:
            yield {"event": "ready", "data": json.dumps({"symbol": symbol.strip().upper()})}
            while not await request.is_disconnected():
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0)
                if message and message.get("type") == "message":
                    payload = message["data"]
                    if not isinstance(payload, str):
                        payload = json.dumps(payload, separators=(",", ":"))
                    event_id = None
                    try:
                        decoded = json.loads(payload)
                        if isinstance(decoded, dict) and decoded.get("event_id") is not None:
                            event_id = str(decoded["event_id"])
                    except (TypeError, ValueError):
                        pass
                    event = {"event": "prediction", "data": payload}
                    if event_id:
                        event["id"] = event_id
                    yield event
                else:
                    await asyncio.sleep(0.1)
        finally:
            try:
                await pubsub.unsubscribe(channel)
            finally:
                await pubsub.aclose()
                await redis_client.aclose()

    return EventSourceResponse(events(), ping=15, send_timeout=30,
                               headers={"Cache-Control": "no-cache"})


@app.get(f"/api/{API_VERSION}/features/events")
async def feature_events(limit: int = Query(500, ge=1, le=5000),
                         _: str = Depends(verify_api_key)):
    """Expose sanitized Kafka feature history for the authenticated dashboard."""
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise HTTPException(status_code=503, detail="Feature event database is not configured")
    try:
        import asyncpg
        connection = await asyncpg.connect(dsn, timeout=5)
        try:
            rows = await connection.fetch(
                """SELECT application_id, applicant_id, event_timestamp,
                          (payload->>'rolling_credit_utilization')::double precision AS rolling_credit_utilization,
                          (payload->>'payment_velocity')::double precision AS payment_velocity,
                          (payload->>'debt_to_income_ratio')::double precision AS debt_to_income_ratio
                   FROM loan_application_audit
                   ORDER BY event_timestamp DESC LIMIT $1""",
                limit,
            )
        finally:
            await connection.close()
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="asyncpg is required for live feature history") from exc
    except Exception as exc:
        LOGGER.exception("Unable to read Kafka feature history")
        raise HTTPException(status_code=503, detail="Feature event history is unavailable") from exc
    return {"events": [dict(row) for row in rows]}


@app.post(f"/api/{API_VERSION}/predict")
def predict(inp: ApplicantInput, _: str = Depends(verify_api_key)):
    try:
        df_in, inc_n, age_n, exp_n = build_input_df(inp)
        prob = score_credit_risk(df_in)

        _, ci_lower, ci_upper = bootstrap_ci(model, df_in, n_bootstrap=100)
        decision   = make_decision(prob)
        confidence = make_confidence(prob)
        gnn_scores = get_gnn_attribution(inp.applicant_id)
        application_context = inp.model_dump()
        if gnn_scores:
            application_context["gnn_scores"] = gnn_scores
        adverse    = generate_adverse_action(
            prob, inc_n, age_n, exp_n, application=application_context
        )

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

        conformal_bounds = calibrated_interval(prob)
        return standardized_risk_response(decision, *(conformal_bounds or (None, None)))
    except Exception:
        LOGGER.exception("Prediction request failed")
        raise HTTPException(status_code=500, detail="Prediction could not be completed")


@app.post(f"/api/{API_VERSION}/predict/batch")
def predict_batch(batch: BatchInput, _: str = Depends(verify_api_key)):
    results = []
    for inp in batch.applicants:
        try:
            df_in, inc_n, age_n, exp_n = build_input_df(inp)
            prob = score_credit_risk(df_in, operation="batch_predict")
            decision = make_decision(prob)
            gnn_scores = get_gnn_attribution(inp.applicant_id)
            pred_id = log_prediction(
                inp.income_lpa, inp.age_years, inp.experience_years,
                inc_n, age_n, exp_n, prob, 0.0, 0.0,
                decision, make_confidence(prob), page="API_BATCH"
            )
            conformal_bounds = calibrated_interval(prob)
            results.append(standardized_risk_response(
                decision, *(conformal_bounds or (None, None))))
        except Exception:
            LOGGER.exception("Batch prediction failed for one applicant")
            results.append(standardized_risk_response("Error"))
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
        with traced_span("model.inference.shap", **{"model.operation": "api_explain",
                                                     "model.name": type(mod).__name__}):
            sv = explainer.shap_values(X)
        shap_vals = sv[1][0] if isinstance(sv, list) else sv[0]
        top = sorted(
            [{"feature": feature_names[i], "shap_value": round(float(shap_vals[i]), 4)}
             for i in range(len(feature_names))],
            key=lambda x: abs(x["shap_value"]), reverse=True
        )[:10]
        gnn_scores = get_gnn_attribution(inp.applicant_id)
        return {
            "top_features": top,
            "base_value": round(float(explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value), 4),
            "graph_risk_attribution": gnn_scores,
            "attribution_sources": {
                "primary_network_signal": ({"source": "Heterogeneous R-GCN", **gnn_scores}
                                           if gnn_scores else None),
                "lightgbm_shap_drivers": top,
            },
        }
    except Exception:
        LOGGER.exception("Explanation request failed")
        raise HTTPException(status_code=500, detail="Explanation could not be generated")


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
