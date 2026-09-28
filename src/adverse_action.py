"""
Adverse Action Notice generator.
Produces regulation-style rejection reasons based on model inputs and SHAP values.
Complies with ECOA (Regulation B §1002.9), FCRA §615(a), RBI IT Framework Sec 6.3,
and DPDP Act 2023 §8.

Two modes:
  1. Plain mode  — original lightweight notice (no external deps).
  2. RAG mode    — enriched notice with inline citations retrieved from the
                   ChromaDB regulatory store via the LangGraph RAG agent.
     Activate by passing use_rag=True (requires chromadb + sentence-transformers).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

REASON_TEMPLATES = {
    "income":      "Insufficient income relative to loan amount",
    "experience":  "Insufficient length of employment history",
    "age":         "Insufficient credit history length (age proxy)",
    "stability":   "Insufficient stability in current residence/employment",
    "income_exp":  "Debt-to-income ratio too high given experience level",
}

# Regulatory references always appended to the plain notice
_PLAIN_CITATION = (
    "Ref: Equal Credit Opportunity Act (ECOA) · Fair Credit Reporting Act (FCRA) · "
    "RBI IT Framework (2023) · DPDP Act 2023"
)


# ── Reason extraction (unchanged logic) ──────────────────────────────────────

def _extract_reasons(
    prob: float,
    income: float,
    age: float,
    experience: float,
    shap_top: list[dict] | None,
) -> list[str]:
    """Derive the 1-3 principal denial reasons from SHAP values or rule-based fallback."""
    reasons: list[str] = []

    if shap_top:
        for item in shap_top[:3]:
            feat = item["feature"].replace("num__", "").replace("cat__", "")
            val  = item["shap_value"]
            if val > 0.01:
                if "Income" in feat:
                    reasons.append(REASON_TEMPLATES["income"])
                elif "Experience" in feat or "JOB" in feat.upper():
                    reasons.append(REASON_TEMPLATES["experience"])
                elif "Age" in feat:
                    reasons.append(REASON_TEMPLATES["age"])
                elif "stability" in feat.lower() or "HOUSE" in feat.upper():
                    reasons.append(REASON_TEMPLATES["stability"])

    if not reasons:
        if income < 0.25:
            reasons.append(REASON_TEMPLATES["income"])
        if experience < 0.15:
            reasons.append(REASON_TEMPLATES["experience"])
        if age < 0.2:
            reasons.append(REASON_TEMPLATES["age"])
        if not reasons:
            reasons.append("Overall risk profile exceeds acceptable threshold")

    return list(dict.fromkeys(reasons))[:3]


# ── Plain notice builder (no external deps) ──────────────────────────────────

def _build_plain_notice(prob: float, reasons: list[str]) -> dict[str, Any]:
    """Build the original structured adverse-action dict (backward compatible)."""
    return {
        "required":   True,
        "decision":   "Application Declined",
        "risk_score": round(prob, 4),
        "reasons":    reasons,
        "notice": (
            "ADVERSE ACTION NOTICE\n"
            "Your loan application has been declined based on information "
            "obtained from our credit risk model. The principal reasons are:\n"
            + "\n".join(f"  {i+1}. {r}" for i, r in enumerate(reasons))
            + "\n\nYou have the right to request the specific reasons for this decision."
        ),
        "citation":   _PLAIN_CITATION,
        # RAG fields default to None so callers can test presence without crashing
        "rag_notice":        None,
        "retrieved_clauses": [],
        "citations_list":    [],
    }


# ── RAG-enriched notice builder ───────────────────────────────────────────────

def _build_rag_notice(
    prob: float,
    reasons: list[str],
    application: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Call the LangGraph RAG agent to produce an inline-cited adverse-action notice.
    Falls back to plain notice on any error so the caller always gets a result.
    """
    try:
        from src.rag_agent import draft_adverse_action_with_citations
        rag_result = draft_adverse_action_with_citations(prob, reasons, application)
        # Merge: keep plain fields, overlay RAG fields
        base = _build_plain_notice(prob, reasons)
        base["rag_notice"]        = rag_result.get("rag_notice") or base["notice"]
        base["retrieved_clauses"] = rag_result.get("retrieved_clauses", [])
        base["citations_list"]    = rag_result.get("citations_list", [])
        # Update citation summary to list retrieved sources
        if base["citations_list"]:
            base["citation"] = "Cited: " + " · ".join(base["citations_list"][:4])
        return base
    except Exception as exc:
        logger.warning("RAG notice generation failed (%s) — falling back to plain notice", exc)
        return _build_plain_notice(prob, reasons)


# ── Public API ────────────────────────────────────────────────────────────────

def generate_adverse_action(
    prob: float,
    income: float,
    age: float,
    experience: float,
    shap_top: list[dict] | None = None,
    use_rag: bool = False,
    application: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Generate a structured adverse-action notice.

    Args:
        prob:        Model predicted default probability (0-1).
        income:      Normalised income (0-1).
        age:         Normalised age (0-1).
        experience:  Normalised experience (0-1).
        shap_top:    Optional list of {"feature": str, "shap_value": float} dicts
                     sorted by absolute SHAP importance.
        use_rag:     If True, enrich the notice with inline regulatory citations
                     retrieved from the ChromaDB regulatory store.
        application: Optional dict with raw application fields (income_lpa, age_years,
                     exp_years, decision …) passed to the RAG agent for richer context.

    Returns:
        Dict with keys:
          required        – bool, whether an adverse-action notice is required
          decision        – str decision label
          risk_score      – float
          reasons         – list[str] principal denial reasons
          notice          – str plain notice (always populated)
          citation        – str regulatory reference line
          rag_notice      – str | None  notice with inline citations (RAG mode only)
          retrieved_clauses – list[dict]  raw clause dicts from ChromaDB
          citations_list  – list[str]   formatted citation strings
    """
    if prob < 0.6:
        return {
            "required":          False,
            "decision":          "Approve / Review",
            "reasons":           [],
            "notice":            "",
            "citation":          "",
            "rag_notice":        None,
            "retrieved_clauses": [],
            "citations_list":    [],
        }

    reasons = _extract_reasons(prob, income, age, experience, shap_top)

    if use_rag:
        return _build_rag_notice(prob, reasons, application)
    return _build_plain_notice(prob, reasons)
