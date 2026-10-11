"""Regression tests for fail-closed regulatory answer validation."""

from src.rag_agent import (
    node_guardrail_fallback,
    node_validate_response,
    route_after_guardrails,
    validate_regulatory_output,
)


def _state(response: str):
    return {
        "intent": "compliance_check",
        "response": response,
        "retrieved_clauses": [{
            "chunk_id": "dpdp-test-chunk",
            "clause_id": "DPDP-Sec12",
            "source": "Digital Personal Data Protection Act 2023 (India)",
            "text": "The Data Principal has the right to correction and erasure of personal data. Erasure requests must be fulfilled within 30 days.",
        }],
    }


def test_unverified_regulatory_claim_routes_to_fallback():
    state = _state("Erasure requests must be fulfilled within 10 days [chunk:dpdp-test-chunk].")

    validated = node_validate_response(state)

    assert validated["guardrail_passed"] is False
    assert any("Statutory limit" in error for error in validated["guardrail_errors"])
    assert route_after_guardrails(validated) == "fallback"

    fallback = node_guardrail_fallback(validated)
    assert fallback["fallback_used"] is True
    assert fallback["response"].startswith("I can’t provide a regulatory conclusion")
    assert "10 days" not in fallback["response"]


def test_unknown_chroma_chunk_id_is_rejected():
    errors = validate_regulatory_output(
        "The Data Principal has a right to correction [chunk:not-retrieved].",
        _state("")["retrieved_clauses"],
    )

    assert any("unknown Chroma chunk" in error for error in errors)


def test_grounded_claim_with_exact_limit_and_chunk_id_passes():
    state = _state("Erasure requests must be fulfilled within 30 days [chunk:dpdp-test-chunk].")

    validated = node_validate_response(state)

    assert validated["guardrail_passed"] is True
    assert route_after_guardrails(validated) == "valid"
