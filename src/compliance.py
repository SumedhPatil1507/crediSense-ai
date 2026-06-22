"""
RBI IT Framework & DPDP Act 2023 Compliance Module.
Documents compliance posture, generates compliance reports,
and provides a credit bureau aggregator interface.

This module covers:
- RBI Master Direction on IT (2023)
- Digital Personal Data Protection Act 2023 (India)
- ECOA / FCRA adverse action requirements
- Basel II Expected Loss framework
"""
import os
import hashlib
import json
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

# ── RBI IT Framework Checklist ─────────────────────────────────────────────────

RBI_CONTROLS = [
    {
        "control_id": "RBI-IT-1",
        "domain": "Governance",
        "requirement": "Board-approved IT policy",
        "status": "Documented",
        "implementation": "config.py + environment-based secrets management",
    },
    {
        "control_id": "RBI-IT-2",
        "domain": "Data Security",
        "requirement": "Encryption of sensitive data at rest and in transit",
        "status": "Implemented",
        "implementation": "AES-256-GCM field-level encryption (src/encryption.py)",
    },
    {
        "control_id": "RBI-IT-3",
        "domain": "Audit Trail",
        "requirement": "Complete audit trail of all transactions",
        "status": "Implemented",
        "implementation": "SHA-256 hashed audit log in SQLite (src/database.py)",
    },
    {
        "control_id": "RBI-IT-4",
        "domain": "Access Control",
        "requirement": "Role-based access control",
        "status": "Partial",
        "implementation": "API key authentication (api/main.py). Production: add RBAC via Auth0/Okta",
    },
    {
        "control_id": "RBI-IT-5",
        "domain": "Model Risk",
        "requirement": "Model validation and documentation",
        "status": "Implemented",
        "implementation": "Model registry with hash verification, performance tracking, drift monitoring",
    },
    {
        "control_id": "RBI-IT-6",
        "domain": "Business Continuity",
        "requirement": "DR/BCP plan for critical systems",
        "status": "Documented",
        "implementation": "Docker containerization, stateless API, SQLite backup",
    },
    {
        "control_id": "RBI-IT-7",
        "domain": "Vendor Risk",
        "requirement": "Third-party risk assessment",
        "status": "Documented",
        "implementation": "Open-source ML stack (scikit-learn, LightGBM). No sensitive data to third parties.",
    },
    {
        "control_id": "RBI-IT-8",
        "domain": "Cyber Security",
        "requirement": "Vulnerability assessment and penetration testing",
        "status": "Partial",
        "implementation": "Input validation (Pydantic), SQL injection prevention (parameterized queries)",
    },
]

# ── DPDP Act 2023 Checklist ────────────────────────────────────────────────────

DPDP_CONTROLS = [
    {
        "control_id": "DPDP-1",
        "principle": "Lawful Processing",
        "requirement": "Obtain valid consent for data processing",
        "status": "Documented",
        "implementation": "Consent required before credit assessment. Adverse action notice provided.",
    },
    {
        "control_id": "DPDP-2",
        "principle": "Purpose Limitation",
        "requirement": "Data used only for stated purpose",
        "status": "Implemented",
        "implementation": "Data used solely for credit risk scoring. No secondary use.",
    },
    {
        "control_id": "DPDP-3",
        "principle": "Data Minimisation",
        "requirement": "Collect only necessary data",
        "status": "Implemented",
        "implementation": "Model uses income, age, experience + contextual fields. No unnecessary PII.",
    },
    {
        "control_id": "DPDP-4",
        "principle": "Accuracy",
        "requirement": "Maintain accurate and up-to-date data",
        "status": "Implemented",
        "implementation": "Input validation + drift monitoring to detect data quality issues.",
    },
    {
        "control_id": "DPDP-5",
        "principle": "Storage Limitation",
        "requirement": "Retain data only as long as necessary",
        "status": "Documented",
        "implementation": "Prediction logs: 90 days retention policy. Implement via scheduled cleanup.",
    },
    {
        "control_id": "DPDP-6",
        "principle": "Right to Access",
        "requirement": "Data principals can access their data",
        "status": "Partial",
        "implementation": "GET /api/v1/predictions/{id} endpoint. Production: add identity verification.",
    },
    {
        "control_id": "DPDP-7",
        "principle": "Right to Erasure",
        "requirement": "Data principals can request deletion",
        "status": "Partial",
        "implementation": "DELETE /api/v1/predictions/{id} stub. Production: implement soft delete + anonymization.",
    },
    {
        "control_id": "DPDP-8",
        "principle": "Security Safeguards",
        "requirement": "Appropriate technical and organisational measures",
        "status": "Implemented",
        "implementation": "AES-256 encryption, PII masking, audit trail, HTTPS enforcement.",
    },
]


def get_compliance_summary() -> dict:
    rbi_implemented = sum(1 for c in RBI_CONTROLS if c["status"] == "Implemented")
    dpdp_implemented = sum(1 for c in DPDP_CONTROLS if c["status"] == "Implemented")
    return {
        "rbi_it_framework": {
            "total": len(RBI_CONTROLS),
            "implemented": rbi_implemented,
            "partial": sum(1 for c in RBI_CONTROLS if c["status"] == "Partial"),
            "compliance_score": f"{rbi_implemented / len(RBI_CONTROLS) * 100:.0f}%",
        },
        "dpdp_act_2023": {
            "total": len(DPDP_CONTROLS),
            "implemented": dpdp_implemented,
            "partial": sum(1 for c in DPDP_CONTROLS if c["status"] == "Partial"),
            "compliance_score": f"{dpdp_implemented / len(DPDP_CONTROLS) * 100:.0f}%",
        },
        "generated_at": datetime.utcnow().isoformat(),
    }


# ── Credit Bureau Aggregator Interface ────────────────────────────────────────

BUREAU_MOCK = os.getenv("BUREAU_MOCK", "true").lower() == "true"
BUREAU_API_KEY = os.getenv("CREDIT_BUREAU_API_KEY", "")
BUREAU_API_URL = os.getenv("CREDIT_BUREAU_URL", "https://api.cibil.com/v1")


def get_bureau_score(pan_hash: str, name_hash: str) -> dict:
    """
    Fetch credit bureau score.
    In mock mode: returns simulated CIBIL-style response.
    In live mode: calls BUREAU_API_URL with API key.

    Note: PAN and name are passed as SHA-256 hashes for privacy.
    The bureau API must support hash-based lookups (anonymized query).
    """
    if BUREAU_MOCK or not BUREAU_API_KEY:
        return _mock_bureau_response(pan_hash)

    try:
        import requests
        resp = requests.post(
            f"{BUREAU_API_URL}/score",
            json={"pan_hash": pan_hash, "name_hash": name_hash},
            headers={"Authorization": f"Bearer {BUREAU_API_KEY}"},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"Bureau API returned {resp.status_code}", "score": None}
    except Exception as e:
        return {"error": str(e), "score": None}


def _mock_bureau_response(pan_hash: str) -> dict:
    """Deterministic mock CIBIL score based on PAN hash."""
    seed = int(pan_hash[:4], 16) % 400
    score = 550 + seed  # Range 550-950
    return {
        "source": "MOCK_CIBIL",
        "score": score,
        "score_band": _cibil_band(score),
        "enquiries_last_6m": int(pan_hash[4:6], 16) % 10,
        "active_loans": int(pan_hash[6:8], 16) % 5,
        "dpd_30_count": 1 if score < 650 else 0,
        "dpd_90_count": 1 if score < 600 else 0,
        "credit_utilization_pct": round((int(pan_hash[8:10], 16) % 80), 1),
        "disclaimer": "MOCK DATA — for development only. Not real bureau data.",
        "timestamp": datetime.utcnow().isoformat(),
    }


def _cibil_band(score: int) -> str:
    if score >= 750:  return "Excellent (750-900)"
    if score >= 700:  return "Good (700-749)"
    if score >= 650:  return "Fair (650-699)"
    if score >= 600:  return "Poor (600-649)"
    return "Very Poor (<600)"


def bureau_to_risk_adjustment(bureau_score: int | None) -> float:
    """
    Convert bureau score to a risk probability adjustment.
    Excellent score reduces ML risk by up to 15%.
    Very poor score increases ML risk by up to 20%.
    """
    if bureau_score is None:
        return 0.0
    if bureau_score >= 750:   return -0.15
    if bureau_score >= 700:   return -0.08
    if bureau_score >= 650:   return -0.03
    if bureau_score >= 600:   return +0.08
    return +0.20
