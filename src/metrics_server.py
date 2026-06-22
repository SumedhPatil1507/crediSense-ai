"""
Prometheus Metrics Integration.
Exposes model performance, request volume, and business metrics.
Works with any Prometheus scraper + Grafana dashboard.

Usage in FastAPI:
    from src.metrics_server import setup_metrics, record_prediction
    setup_metrics(app)  # adds /metrics endpoint
    record_prediction(prob, decision, latency_ms)

Grafana dashboard JSON is in infra/grafana_dashboard.json.
"""
import time
import os

PROMETHEUS_ENABLED = os.getenv("PROMETHEUS_ENABLED", "false").lower() == "true"

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary,
        generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
    )
    PROM_AVAILABLE = True
except ImportError:
    PROM_AVAILABLE = False

# ── Metric Definitions ─────────────────────────────────────────────────────────

if PROM_AVAILABLE:
    PREDICTION_COUNTER = Counter(
        "credisense_predictions_total",
        "Total number of predictions made",
        ["decision", "confidence", "page"]
    )
    PREDICTION_LATENCY = Histogram(
        "credisense_prediction_latency_ms",
        "Prediction latency in milliseconds",
        buckets=[10, 25, 50, 100, 250, 500, 1000, 2500]
    )
    RISK_SCORE_HISTOGRAM = Histogram(
        "credisense_risk_score",
        "Distribution of risk probability scores",
        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    HITL_QUEUE_GAUGE = Gauge(
        "credisense_hitl_queue_pending",
        "Number of cases pending in HITL queue"
    )
    MODEL_PSI_GAUGE = Gauge(
        "credisense_model_psi",
        "Current Population Stability Index (model drift)"
    )
    BUREAU_REQUESTS = Counter(
        "credisense_bureau_requests_total",
        "Total credit bureau API requests",
        ["source", "status"]
    )
    FEEDBACK_COUNTER = Counter(
        "credisense_feedback_total",
        "Analyst feedback submissions",
        ["feedback_type"]
    )
    API_ERROR_COUNTER = Counter(
        "credisense_api_errors_total",
        "API errors by endpoint",
        ["endpoint", "status_code"]
    )


def record_prediction(prob: float, decision: str, confidence: str,
                       latency_ms: float, page: str = "API"):
    if not PROM_AVAILABLE or not PROMETHEUS_ENABLED:
        return
    PREDICTION_COUNTER.labels(decision=decision, confidence=confidence, page=page).inc()
    PREDICTION_LATENCY.observe(latency_ms)
    RISK_SCORE_HISTOGRAM.observe(prob)


def update_hitl_queue_gauge(pending: int):
    if PROM_AVAILABLE and PROMETHEUS_ENABLED:
        HITL_QUEUE_GAUGE.set(pending)


def update_psi_gauge(psi: float):
    if PROM_AVAILABLE and PROMETHEUS_ENABLED:
        MODEL_PSI_GAUGE.set(psi)


def record_feedback(feedback_type: str):
    if PROM_AVAILABLE and PROMETHEUS_ENABLED:
        FEEDBACK_COUNTER.labels(feedback_type=feedback_type).inc()


def record_api_error(endpoint: str, status_code: int):
    if PROM_AVAILABLE and PROMETHEUS_ENABLED:
        API_ERROR_COUNTER.labels(endpoint=endpoint, status_code=str(status_code)).inc()


def setup_metrics(app):
    """Add /metrics endpoint to FastAPI app."""
    if not PROM_AVAILABLE or not PROMETHEUS_ENABLED:
        return

    from fastapi import Response

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def metrics_status() -> dict:
    return {
        "prometheus_available": PROM_AVAILABLE,
        "prometheus_enabled": PROMETHEUS_ENABLED,
        "metrics_endpoint": "/metrics" if (PROM_AVAILABLE and PROMETHEUS_ENABLED) else None,
    }
