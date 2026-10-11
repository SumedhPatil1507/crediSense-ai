"""Run a 50-query RAGAS audit against CrediSense's regulatory copilot.

Example: ``python -m eval.ragas_suite --output eval/reports/ragas_audit.json``
The references intentionally come from the bundled seed corpus; review that
corpus against primary regulatory publications before treating scores as legal
assurance.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.rag_ingest import SEED_CLAUSES
from src.rag_agent import run_compliance_query


TOPICS = [
    ("RBI-IT-Sec2.1", "RBI IT governance responsibilities"),
    ("RBI-IT-Sec3.1", "RBI IT risk management"),
    ("RBI-IT-Sec4.2", "RBI information security controls"),
    ("RBI-IT-Sec5.1", "RBI audit and assurance"),
    ("RBI-IT-Sec6.3", "RBI model governance"),
    ("DPDP-Sec6", "DPDP consent requirements"),
    ("DPDP-Sec8", "DPDP data fiduciary duties"),
    ("DPDP-Sec9", "DPDP processing of children's data"),
    ("DPDP-Sec12", "DPDP correction and erasure rights"),
    ("DPDP-Sec14", "DPDP grievance redressal"),
]
QUESTION_FORMS = [
    "What does the source say about {topic}?",
    "Summarize the requirements for {topic}.",
    "Which obligations are stated for {topic}?",
    "What should a compliance reviewer check for {topic}?",
    "What evidence in the cited material addresses {topic}?",
]


def build_queries() -> list[dict[str, str]]:
    """Build exactly 50 query/reference pairs from seeded clause text."""
    by_id = {clause["clause_id"]: clause for clause in SEED_CLAUSES}
    missing = [clause_id for clause_id, _ in TOPICS if clause_id not in by_id]
    if missing:
        raise RuntimeError(f"RAGAS topics reference missing seed clauses: {missing}")
    rows = []
    for clause_id, topic in TOPICS:
        reference = by_id[clause_id]["text"]
        for question_form in QUESTION_FORMS:
            rows.append({"user_input": question_form.format(topic=topic), "reference": reference})
    if len(rows) != 50:
        raise AssertionError(f"Expected 50 queries, got {len(rows)}")
    return rows


def collect_responses(queries: list[dict[str, str]]) -> list[dict[str, Any]]:
    dataset = []
    for item in queries:
        result = run_compliance_query(item["user_input"])
        clauses = result.get("retrieved_clauses", [])
        dataset.append({
            **item,
            "response": result.get("response", ""),
            "retrieved_contexts": [clause.get("text", "") for clause in clauses],
            "retrieved_context_ids": [clause.get("chunk_id", "") for clause in clauses],
            "guardrail_passed": bool(result.get("guardrail_passed", False)),
            "fallback_used": bool(result.get("fallback_used", False)),
            "guardrail_errors": result.get("guardrail_errors", []),
        })
    return dataset


def evaluate_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY for the RAGAS evaluator (and copilot generation).")
    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install evaluation dependencies with `pip install -r requirements-rag-eval.txt`."
        ) from exc

    evaluator = LangchainLLMWrapper(ChatOpenAI(model=os.getenv("RAGAS_MODEL", "gpt-4o-mini"), temperature=0))
    metrics = [
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
        Faithfulness(),
        ResponseRelevancy(),
    ]
    columns = ["user_input", "response", "retrieved_contexts", "reference"]
    result = evaluate(Dataset.from_list([{key: row.get(key) for key in columns} for row in rows]), metrics=metrics, llm=evaluator)
    try:
        detail_rows = result.to_pandas().to_dict(orient="records")
        details = [{k: _json_value(v) for k, v in row.items()} for row in detail_rows]
    except (AttributeError, TypeError):
        details = []
    summary = {k: _json_value(v) for k, v in result.items()}
    return summary, details


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("eval/reports/ragas_audit.json"))
    args = parser.parse_args()

    rows = collect_responses(build_queries())
    summary, metric_rows = evaluate_rows(rows)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite": "credisense-regulatory-rag-50",
        "query_count": len(rows),
        "corpus": "src.rag_ingest.SEED_CLAUSES",
        "corpus_validation_note": "Metrics measure agreement with bundled seed text; they do not certify statutory accuracy or currency.",
        "metrics": summary,
        "rows": [
            {**row, "ragas_metrics": metric_rows[i] if i < len(metric_rows) else {}}
            for i, row in enumerate(rows)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(rows)}-query RAGAS audit to {args.output}")


if __name__ == "__main__":
    main()
