"""
Regulatory RAG Copilot — LangGraph Agent.

Architecture:
  User Query / Rejected Application
       │
       ▼
  [intent_router]  ── classifies: adverse_action | compliance_check | general_query
       │
       ├─► [retrieve_clauses]  ── ChromaDB + cross-encoder reranker
       │         │
       ├─► [draft_notice]      ── adverse-action path: generates ECOA/FCRA notice with citations
       │
       └─► [answer_question]   ── compliance / general: cited Q&A answer
       │
       ▼
  [format_response]  ── standardised output with inline citations

LLM Backend:
- Primary:  OpenAI GPT-4o-mini (via OPENAI_API_KEY env var)
- Fallback:  Groq Llama-3.1-8B-instant (via GROQ_API_KEY)
- Offline:  Template-based generator (no API key needed) using retrieved clauses

The agent is intentionally LLM-agnostic; any LangChain-compatible chat model works.
"""

from __future__ import annotations

import logging
import os
from typing import Any, TypedDict

logger = logging.getLogger(__name__)

# ── State Definition ──────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    """LangGraph state shared across all nodes."""
    # Input
    query: str
    application: dict[str, Any]   # rejected application details (optional)
    denial_reasons: list[str]     # SHAP-derived reasons (optional)
    risk_score: float

    # Routing
    intent: str                   # "adverse_action" | "compliance_check" | "general_query"

    # Retrieval
    retrieved_clauses: list[dict[str, Any]]
    context_block: str

    # Output
    response: str
    citations: list[str]
    notice: str                   # formatted adverse-action notice (adverse_action intent only)
    error: str


# ── LLM Loader ───────────────────────────────────────────────────────────────

def _load_llm():
    """Load the best available LLM. Returns (llm, provider_name) or (None, 'offline')."""
    # Try OpenAI
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key and openai_key != "your-openai-api-key-here":
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens=1500,
                api_key=openai_key,
            )
            logger.info("LLM: OpenAI gpt-4o-mini")
            return llm, "openai"
        except ImportError:
            logger.debug("langchain-openai not installed")
        except Exception as e:
            logger.warning("OpenAI init failed: %s", e)

    # Try Groq
    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key and groq_key != "your-groq-api-key-here":
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=1500,
                groq_api_key=groq_key,
            )
            logger.info("LLM: Groq llama-3.1-8b-instant")
            return llm, "groq"
        except ImportError:
            logger.debug("langchain-groq not installed")
        except Exception as e:
            logger.warning("Groq init failed: %s", e)

    logger.info("LLM: Offline template mode (no API key configured)")
    return None, "offline"


# ── Prompt Templates ─────────────────────────────────────────────────────────

_ADVERSE_ACTION_SYSTEM = """You are CrediSense Regulatory Copilot, an expert in credit regulation compliance.
Your task is to draft a formal Adverse Action Notice that complies with ECOA (Regulation B), FCRA, RBI IT Framework, and DPDP Act 2023.

RULES:
1. Structure the notice exactly as shown in the TEMPLATE section.
2. After EVERY substantive statement, add an inline citation in the format [Source · ClauseID].
3. Use ONLY the regulatory text provided in CONTEXT. Do not invent clauses.
4. List the specific denial reasons with their regulatory basis.
5. Include the applicant's rights section citing the specific regulation that grants each right.
6. Be formal, precise, and legally accurate.
"""

_ADVERSE_ACTION_HUMAN = """
APPLICATION DETAILS:
Risk Score: {risk_score}
Decision: Application Declined
Denial Reasons:
{reasons_block}

REGULATORY CONTEXT (retrieved clauses):
{context_block}

TEMPLATE:
---
ADVERSE ACTION NOTICE

Applicant Reference: [ID]
Decision Date: [DATE]
Decision: APPLICATION DECLINED
Risk Score: {risk_score}

PRINCIPAL REASONS FOR ADVERSE ACTION:
[List each reason with regulatory basis and inline citation]

YOUR RIGHTS UNDER APPLICABLE LAW:
[List rights with inline citations to specific clauses]

CONTACT INFORMATION:
[Details for disputes]

REGULATORY REFERENCES:
[List all cited regulations]
---

Draft the notice now, populating the template with the application details and citing specific clauses inline.
"""

_COMPLIANCE_SYSTEM = """You are CrediSense Regulatory Copilot, an expert compliance analyst specialising in:
- RBI Master Direction on IT Framework (2023)
- Digital Personal Data Protection Act 2023 (India)
- Equal Credit Opportunity Act / Regulation B (ECOA)
- Fair Credit Reporting Act (FCRA)

Your job is to answer compliance questions with precise citations to specific regulatory clauses.

RULES:
1. Answer directly and concisely.
2. Cite EVERY regulatory claim inline using the format [Source · ClauseID].
3. Use ONLY the regulatory text in CONTEXT. Clearly state if something is not covered.
4. Structure answers as: Direct Answer → Regulatory Basis → Specific Requirements → Gaps/Risks.
5. Flag any compliance gaps explicitly as ⚠️ COMPLIANCE GAP.
"""

_COMPLIANCE_HUMAN = """
ANALYST QUESTION:
{query}

APPLICATION CONTEXT (if relevant):
{app_context}

REGULATORY CONTEXT (retrieved clauses):
{context_block}

Provide a cited compliance analysis. For each claim, cite the specific clause from the context above.
"""


# ── Node Functions ────────────────────────────────────────────────────────────

def node_intent_router(state: AgentState) -> AgentState:
    """Classify the query intent without calling the LLM (fast keyword routing)."""
    query = (state.get("query") or "").lower()
    denial_reasons = state.get("denial_reasons", [])

    # Explicit adverse-action trigger
    adverse_keywords = {
        "adverse action", "decline", "reject", "denial", "denied", "refused",
        "draft notice", "adverse notice", "ecoa", "fcra", "action notice",
    }
    compliance_keywords = {
        "compliant", "compliance", "violat", "legal", "regulation", "rbi",
        "dpdp", "gdpr", "audit", "breach", "lawful", "permission", "consent",
        "data protection", "right to", "obligation",
    }

    if denial_reasons or any(kw in query for kw in adverse_keywords):
        intent = "adverse_action"
    elif any(kw in query for kw in compliance_keywords):
        intent = "compliance_check"
    else:
        intent = "general_query"

    logger.debug("Intent classified: %s", intent)
    return {**state, "intent": intent}


def node_retrieve_clauses(state: AgentState) -> AgentState:
    """Retrieve relevant regulatory clauses from ChromaDB + cross-encoder rerank."""
    from src.rag_retriever import get_retriever

    retriever = get_retriever(n_candidates=20, top_n=6)
    intent = state.get("intent", "general_query")
    query = state.get("query", "")
    denial_reasons = state.get("denial_reasons", [])

    try:
        if intent == "adverse_action" and denial_reasons:
            clauses = retriever.retrieve_for_adverse_action(denial_reasons)
            # Also retrieve on the query text if provided
            if query:
                extra = retriever.retrieve(query)
                # Merge, deduplicate by clause_id
                seen = {c.clause_id for c in clauses}
                for c in extra:
                    if c.clause_id not in seen:
                        clauses.append(c)
                        seen.add(c.clause_id)
                clauses = sorted(clauses, key=lambda x: x.cross_score, reverse=True)[:6]
        else:
            clauses = retriever.retrieve(
                query or "credit risk adverse action compliance requirements"
            )

        context_block = retriever.format_context_block(clauses)
        retrieved_dicts = [c.to_dict() for c in clauses]
        citations = list({c.citation for c in clauses})

        logger.debug("Retrieved %d clauses", len(clauses))
        return {
            **state,
            "retrieved_clauses": retrieved_dicts,
            "context_block": context_block,
            "citations": citations,
        }
    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        return {
            **state,
            "retrieved_clauses": [],
            "context_block": "No regulatory context available.",
            "citations": [],
            "error": f"Retrieval error: {e}",
        }


def node_generate_response(state: AgentState) -> AgentState:
    """Generate the final response (LLM or template fallback)."""
    intent = state.get("intent", "general_query")
    context_block = state.get("context_block", "")
    query = state.get("query", "")
    denial_reasons = state.get("denial_reasons", [])
    risk_score = state.get("risk_score", 0.0)
    application = state.get("application", {})
    retrieved_clauses = state.get("retrieved_clauses", [])

    llm, provider = _load_llm()

    # ── Adverse Action Path ────────────────────────────────────────────────
    if intent == "adverse_action":
        reasons_block = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(denial_reasons))

        if llm is not None:
            try:
                from langchain_core.messages import HumanMessage, SystemMessage
                messages = [
                    SystemMessage(content=_ADVERSE_ACTION_SYSTEM),
                    HumanMessage(content=_ADVERSE_ACTION_HUMAN.format(
                        risk_score=f"{risk_score:.2%}",
                        reasons_block=reasons_block,
                        context_block=context_block,
                    )),
                ]
                result = llm.invoke(messages)
                notice = result.content
                return {**state, "notice": notice, "response": notice}
            except Exception as e:
                logger.error("LLM call failed: %s — using template fallback", e)

        # Template fallback
        notice = _template_adverse_action(
            risk_score, denial_reasons, retrieved_clauses
        )
        return {**state, "notice": notice, "response": notice}

    # ── Compliance / General Path ──────────────────────────────────────────
    app_context = ""
    if application:
        app_context = (
            f"Income: {application.get('income_lpa', 'N/A')} LPA | "
            f"Age: {application.get('age_years', 'N/A')} years | "
            f"Experience: {application.get('exp_years', 'N/A')} years | "
            f"Risk Score: {risk_score:.2%} | "
            f"Decision: {application.get('decision', 'N/A')}"
        )

    if llm is not None:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content=_COMPLIANCE_SYSTEM),
                HumanMessage(content=_COMPLIANCE_HUMAN.format(
                    query=query,
                    app_context=app_context or "No specific application provided.",
                    context_block=context_block,
                )),
            ]
            result = llm.invoke(messages)
            response = result.content
            return {**state, "response": response}
        except Exception as e:
            logger.error("LLM call failed: %s — using template fallback", e)

    # Template fallback
    response = _template_compliance_answer(query, retrieved_clauses)
    return {**state, "response": response}


# ── Template Fallbacks (no LLM required) ─────────────────────────────────────

def _template_adverse_action(
    risk_score: float,
    reasons: list[str],
    clauses: list[dict[str, Any]],
) -> str:
    """Generate a structured adverse action notice using retrieved clauses as citations."""
    from datetime import datetime

    # Map reasons to relevant clauses
    ecoa_clauses = [c for c in clauses if "ECOA" in c.get("source", "") or "1002" in c.get("source", "")]
    fcra_clauses = [c for c in clauses if "FCRA" in c.get("source", "") or "1681" in c.get("source", "")]
    rbi_clauses  = [c for c in clauses if "RBI" in c.get("source", "")]
    dpdp_clauses = [c for c in clauses if "DPDP" in c.get("source", "") or "Digital Personal" in c.get("source", "")]

    ecoa_cit  = ecoa_clauses[0]["citation"]  if ecoa_clauses  else "ECOA Regulation B §1002.9"
    ecoa9b    = next((c["citation"] for c in ecoa_clauses if "9b" in c.get("clause_id","").lower() or "9(b)" in c.get("text","")), ecoa_cit)
    fcra_cit  = fcra_clauses[0]["citation"]  if fcra_clauses  else "FCRA §615(a)"
    rbi_cit   = rbi_clauses[0]["citation"]   if rbi_clauses   else "RBI IT Framework Sec 6.3"
    dpdp_cit  = dpdp_clauses[0]["citation"]  if dpdp_clauses  else "DPDP Act 2023 §8"

    reasons_block = "\n".join(
        f"  {i+1}. {r}\n      Regulatory basis: [{ecoa9b}]"
        for i, r in enumerate(reasons)
    )

    all_citations = "\n".join(
        f"  • {c['citation']}"
        for c in clauses[:6]
        if c.get("citation")
    )

    return f"""ADVERSE ACTION NOTICE
{'='*60}
Decision Date: {datetime.utcnow().strftime('%Y-%m-%d')}
Decision:      APPLICATION DECLINED
Risk Score:    {risk_score:.2%}

PRINCIPAL REASONS FOR ADVERSE ACTION
[As required by {ecoa_cit}]

{reasons_block}

YOUR RIGHTS UNDER APPLICABLE LAW

1. Right to Statement of Reasons
   You have the right to request a specific statement of the reasons for
   this decision within 60 days of receiving this notice.
   [{ecoa_cit}]

2. Right to Credit Report
   If a consumer report was used in this decision, you have the right to
   obtain a free copy of your report within 60 days and to dispute any
   inaccurate information.
   [{fcra_cit}]

3. Right to Data Access (India)
   As a data principal, you have the right to access and correct personal
   data processed in connection with this decision.
   [{dpdp_cit}]

4. Model Governance
   This decision was made using a validated algorithmic model subject to
   periodic review and audit in accordance with applicable IT governance
   requirements.
   [{rbi_cit}]

DISPUTE / CONTACT
To request a statement of reasons or dispute this decision, contact:
CrediSense Credit Operations | compliance@credisense.ai

REGULATORY REFERENCES CITED
{all_citations}

{'='*60}
This notice is generated in accordance with the Equal Credit Opportunity
Act (ECOA), Fair Credit Reporting Act (FCRA), RBI IT Framework (2023),
and Digital Personal Data Protection Act 2023.
"""


def _template_compliance_answer(query: str, clauses: list[dict[str, Any]]) -> str:
    """Generate a cited compliance answer from retrieved clauses."""
    if not clauses:
        # Detect whether the query is out-of-scope vs. just an empty store
        _OUT_OF_SCOPE_HINTS = {
            "withdraw": "transaction monitoring / AML",
            "withdrawal": "transaction monitoring / AML",
            "deposit": "banking transactions / AML",
            "transfer": "fund transfers / payment systems",
            "cash": "cash transaction reporting (CTR/AML)",
            "aml": "Anti-Money Laundering (PMLA / FATF)",
            "money laundering": "Anti-Money Laundering",
            "sanctions": "sanctions screening",
            "kyc": "KYC / Customer Due Diligence",
            "suspicious": "suspicious transaction reporting (STR)",
            "fraud": "transaction fraud detection",
            "upi": "UPI / payment rails",
            "neft": "NEFT / RTGS payment systems",
            "tax": "taxation / TDS",
            "interest rate": "RBI monetary policy",
            "stock": "securities / SEBI",
            "investment": "wealth management / SEBI",
            "insurance": "insurance regulation / IRDAI",
        }
        q_lower = query.lower()
        matched_domain = next(
            (domain for hint, domain in _OUT_OF_SCOPE_HINTS.items() if hint in q_lower),
            None,
        )

        if matched_domain:
            return (
                f"This question is outside the scope of the CrediSense Regulatory Copilot.\n\n"
                f"**Why:** The query appears to be about **{matched_domain}**, which is not "
                f"covered by the knowledge base in this system.\n\n"
                f"**What this copilot covers:**\n"
                f"- ECOA Regulation B — adverse action notices, credit decision fairness\n"
                f"- FCRA — consumer report rights, dispute process\n"
                f"- RBI IT Framework (2023) — model governance, encryption, audit trails, BCP\n"
                f"- DPDP Act 2023 — personal data consent, retention, erasure, breach notification\n\n"
                f"**Try asking:**\n"
                f"- \"Is this credit rejection compliant with ECOA?\"\n"
                f"- \"What does the DPDP Act require for data retention after a loan decision?\"\n"
                f"- \"Draft an adverse action notice for this rejected application.\"\n"
                f"- \"What are the RBI IT Framework audit trail requirements?\""
            )

        # Store likely empty — generic init message
        return (
            "No directly relevant regulatory clauses were retrieved for this query.\n\n"
            "This can happen if:\n"
            "1. The regulatory store hasn't been initialised yet — run "
            "`python -m src.ingest_regulations` to load the seed knowledge base.\n"
            "2. The question is phrased in a way that doesn't match the regulatory text — "
            "try rephrasing around specific obligations, rights, or compliance requirements.\n\n"
            "**In-scope topics:** ECOA adverse action notices · FCRA consumer rights · "
            "RBI IT Framework model governance · DPDP Act 2023 data protection."
        )

    lines = [
        f"**Compliance Analysis** — Based on {len(clauses)} retrieved regulatory clause(s):\n"
    ]

    for c in clauses[:4]:
        lines.append(
            f"**[{c['rank']}] {c['citation']}**\n"
            f"{c['text'][:400]}{'...' if len(c['text']) > 400 else ''}\n"
        )

    lines.append(
        "\n*Note: For a full LLM-powered analysis with synthesised reasoning, "
        "configure OPENAI_API_KEY or GROQ_API_KEY in your environment.*"
    )

    return "\n\n".join(lines)


# ── LangGraph Graph Builder ───────────────────────────────────────────────────

def _build_graph():
    """Build the LangGraph StateGraph. Returns compiled graph or None."""
    try:
        from langgraph.graph import StateGraph, END

        graph = StateGraph(AgentState)

        graph.add_node("intent_router",    node_intent_router)
        graph.add_node("retrieve_clauses", node_retrieve_clauses)
        graph.add_node("generate",         node_generate_response)

        graph.set_entry_point("intent_router")
        graph.add_edge("intent_router",    "retrieve_clauses")
        graph.add_edge("retrieve_clauses", "generate")
        graph.add_edge("generate",          END)

        return graph.compile()
    except ImportError:
        logger.warning("langgraph not installed — using sequential fallback")
        return None


_compiled_graph = None
_graph_lock = __import__("threading").Lock()


def get_graph():
    """Lazily compile and cache the LangGraph."""
    global _compiled_graph
    if _compiled_graph is None:
        with _graph_lock:
            if _compiled_graph is None:
                _compiled_graph = _build_graph()
    return _compiled_graph


# ── Public API ────────────────────────────────────────────────────────────────

def run_compliance_query(
    query: str,
    application: dict[str, Any] | None = None,
    denial_reasons: list[str] | None = None,
    risk_score: float = 0.0,
) -> AgentState:
    """
    Run the regulatory RAG agent on a compliance question or adverse-action request.

    Args:
        query:          Analyst's natural language question.
        application:    Dict with income_lpa, age_years, exp_years, decision etc.
        denial_reasons: List of SHAP-based denial reasons (triggers adverse-action path).
        risk_score:     Model predicted default probability.

    Returns:
        AgentState with .response, .notice, .citations, .retrieved_clauses populated.
    """
    initial: AgentState = {
        "query": query,
        "application": application or {},
        "denial_reasons": denial_reasons or [],
        "risk_score": risk_score,
        "intent": "",
        "retrieved_clauses": [],
        "context_block": "",
        "response": "",
        "citations": [],
        "notice": "",
        "error": "",
    }

    graph = get_graph()
    if graph is not None:
        try:
            final_state = graph.invoke(initial)
            return final_state
        except Exception as e:
            logger.error("LangGraph execution failed: %s", e)
            # Fall through to sequential fallback

    # Sequential fallback (no langgraph)
    state = node_intent_router(initial)
    state = node_retrieve_clauses(state)
    state = node_generate_response(state)
    return state


def draft_adverse_action_with_citations(
    prob: float,
    reasons: list[str],
    application: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    High-level function called by adverse_action.py.
    Returns a dict compatible with generate_adverse_action() output, enriched with
    inline regulatory citations.

    Args:
        prob:        Default probability from the model.
        reasons:     SHAP-derived denial reasons.
        application: Optional application details for context.

    Returns:
        Dict with keys: required, decision, risk_score, reasons, notice, citation,
                        rag_notice, retrieved_clauses, citations_list
    """
    result = run_compliance_query(
        query="draft adverse action notice with regulatory citations",
        application=application,
        denial_reasons=reasons,
        risk_score=prob,
    )

    return {
        "required": True,
        "decision": "Application Declined",
        "risk_score": round(prob, 4),
        "reasons": reasons,
        # Original plain notice (kept for backward compat)
        "notice": (
            "ADVERSE ACTION NOTICE\n"
            "Your loan application has been declined based on information "
            "obtained from our credit risk model. The principal reasons are:\n"
            + "\n".join(f"  {i+1}. {r}" for i, r in enumerate(reasons))
            + "\n\nYou have the right to request the specific reasons for this decision."
        ),
        "citation": "Ref: Equal Credit Opportunity Act (ECOA) · Fair Credit Reporting Act (FCRA)",
        # RAG-enriched notice with inline citations
        "rag_notice": result.get("notice") or result.get("response", ""),
        "retrieved_clauses": result.get("retrieved_clauses", []),
        "citations_list": result.get("citations", []),
    }
