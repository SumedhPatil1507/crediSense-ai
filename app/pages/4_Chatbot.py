"""
AI Risk Assistant — Chatbot page.

Tab 1: Risk Prediction   — ML scoring + adverse-action notice (with optional RAG citations)
Tab 2: Regulatory Copilot — Live RAG chat: analyst asks compliance questions,
                             gets cited answers backed by ChromaDB regulatory store
                             (RBI IT Framework, DPDP Act 2023, ECOA/FCRA).
"""

import streamlit as st
import joblib
import json
import warnings
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from utils import build_full_input
from src.database import log_prediction, log_feedback
from src.confidence_intervals import bootstrap_ci
from src.adverse_action import generate_adverse_action
from src.hitl_queue import enqueue as hitl_enqueue
from src.config import THRESHOLD_APPROVE, THRESHOLD_REVIEW

model = joblib.load(BASE_DIR / "models/model.pkl")
with open(BASE_DIR / "models/columns.json") as f:
    cols = json.load(f)


def normalize(income_lpa, age_years, exp_years):
    return (
        min(income_lpa / 50.0, 1.0),
        (age_years - 18) / 52.0,
        min(exp_years / 40.0, 1.0),
    )


# ── RAG helpers (lazy-imported so page loads without deps) ──────────────────

@st.cache_resource(show_spinner="Loading regulatory knowledge base…")
def _load_rag_agent():
    """Load and warm up the RAG agent once per session."""
    try:
        from src.rag_retriever import get_retriever
        from src.rag_ingest import ingest_all_pdfs, get_collection_stats

        # Auto-ingest seed knowledge if collection is empty
        stats = get_collection_stats()
        if stats.get("total_chunks", 0) == 0:
            ingest_all_pdfs()

        retriever = get_retriever(n_candidates=20, top_n=5)
        return retriever, True
    except ImportError as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)


def _run_rag_query(query: str, application: dict | None = None,
                   denial_reasons: list | None = None,
                   risk_score: float = 0.0) -> dict:
    """Run the LangGraph RAG agent and return the result dict."""
    from src.rag_agent import run_compliance_query
    return run_compliance_query(
        query=query,
        application=application,
        denial_reasons=denial_reasons or [],
        risk_score=risk_score,
    )


# ── Suggested starter questions ─────────────────────────────────────────────

_STARTER_QUESTIONS = [
    "Is this rejection compliant with ECOA Regulation B?",
    "What does the DPDP Act say about retaining credit decision data?",
    "Draft an adverse action notice for the last rejected application.",
    "What are the RBI IT Framework requirements for algorithmic credit models?",
    "Does FCRA §615(a) require a free credit report offer after rejection?",
    "What consent obligations apply under DPDP Act §4 for credit processing?",
    "Summarise the data encryption requirements under RBI IT Sec 4.2.",
    "What are the audit trail obligations for credit decisions under RBI IT Sec 5.1?",
]


# ── Page Layout ──────────────────────────────────────────────────────────────

st.set_page_config(layout="wide")
st.title("AI Risk Assistant")

tabs = st.tabs(["Risk Prediction", "Regulatory Copilot"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Risk Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    with c1:
        income_lpa = st.number_input("Annual Income (LPA)", min_value=0.5, max_value=500.0,
                                      value=8.0, step=0.5, key="cb_income")
        profession = st.selectbox("Profession",
                                  ["Engineer", "Doctor", "Lawyer", "Teacher",
                                   "Accountant", "Manager", "Analyst", "Other"],
                                  key="cb_profession")
    with c2:
        age_years  = st.number_input("Age (years)", min_value=18, max_value=70,
                                      value=30, key="cb_age")
        house_own  = st.selectbox("House Ownership",
                                  ["owned", "rented", "norent_noown"], key="cb_house")
    with c3:
        exp_years  = st.number_input("Work Experience (years)", min_value=0, max_value=45,
                                      value=5, key="cb_exp")
        marital    = st.selectbox("Marital Status", ["single", "married"], key="cb_marital")

    use_rag_notice = st.checkbox(
        "Enrich adverse-action notice with inline regulatory citations (RAG)",
        value=True,
        help="Retrieves specific ECOA/FCRA/RBI/DPDP clauses from the vector store and inlines them.",
    )

    if st.button("Assess Risk", use_container_width=True, type="primary", key="cb_predict"):
        if exp_years >= age_years - 16:
            st.error("Experience cannot exceed working age.")
            st.stop()

        income_n, age_n, exp_n = normalize(income_lpa, age_years, exp_years)
        age_group = "Young" if age_n < 0.3 else "Senior" if age_n > 0.7 else "Middle"

        user_input = {
            "Income": income_n, "Age": age_n, "Experience": exp_n,
            "CURRENT_JOB_YRS": 2, "CURRENT_HOUSE_YRS": 3,
            "House_Ownership": house_own, "Married/Single": marital,
            "Car_Ownership": "no", "Profession": profession,
            "CITY": "Mumbai", "STATE": "Maharashtra", "age_group": age_group,
        }
        df_in = build_full_input(user_input, cols)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob = float(model.predict_proba(df_in)[0][1])

        with st.spinner("Computing confidence interval…"):
            _, ci_lower, ci_upper = bootstrap_ci(model, df_in, n_bootstrap=100)

        decision   = ("Approve" if prob < THRESHOLD_APPROVE
                      else "Manual Review" if prob < THRESHOLD_REVIEW
                      else "Reject")
        margin     = abs(prob - 0.5)
        confidence = ("High" if margin > 0.3
                      else "Medium" if margin > 0.15
                      else "Low (borderline)")

        st.session_state["chatbot_pred"] = dict(
            income_lpa=income_lpa, age_years=age_years, exp_years=exp_years,
            income_n=income_n, age_n=age_n, exp_n=exp_n,
            prob=prob, decision=decision, confidence=confidence,
        )

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Default Risk",     f"{prob:.2%}")
        r2.metric("Safe Probability", f"{1-prob:.2%}")
        r3.metric("Confidence",       confidence)
        r4.metric("Decision",         decision)
        st.caption(f"95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]")

        if prob > THRESHOLD_REVIEW:
            st.error(f"High Default Risk: {prob:.2%}")
        elif prob > THRESHOLD_APPROVE:
            st.warning(f"Moderate Risk: {prob:.2%}")
        else:
            st.success(f"Low Risk: {prob:.2%}")

        drivers = []
        if income_lpa < 5:             drivers.append(f"low income ({income_lpa} LPA)")
        if exp_years < 2:              drivers.append("limited work experience")
        if age_years < 25:             drivers.append("young applicant")
        if income_lpa > 20 and exp_years > 8:
            drivers.append("strong income and experience")
        driver_text = f" Key factors: {', '.join(drivers)}." if drivers else ""
        risk_label  = ("low" if prob < THRESHOLD_APPROVE
                       else "moderate" if prob < THRESHOLD_REVIEW else "high")
        st.info(
            f"This applicant has a {risk_label} default risk ({prob:.1%}).{driver_text} "
            f"Recommendation: {decision}."
        )

        # ── Adverse Action Notice ────────────────────────────────────────
        application_ctx = dict(
            income_lpa=income_lpa, age_years=age_years, exp_years=exp_years,
            decision=decision,
        )
        adverse = generate_adverse_action(
            prob, income_n, age_n, exp_n,
            use_rag=use_rag_notice,
            application=application_ctx,
        )

        if adverse["required"]:
            # Display RAG notice if available, fall back to plain
            displayed_notice = adverse.get("rag_notice") or adverse["notice"]

            with st.expander("📋 Adverse Action Notice", expanded=True):
                st.text(displayed_notice)

                if adverse.get("citations_list"):
                    st.markdown("**Regulatory clauses cited:**")
                    for cit in adverse["citations_list"]:
                        st.markdown(f"- {cit}")

                if adverse.get("retrieved_clauses"):
                    with st.expander("View retrieved regulatory clauses", expanded=False):
                        for clause in adverse["retrieved_clauses"][:4]:
                            st.markdown(
                                f"**[{clause['rank']}] {clause['citation']}** "
                                f"*(relevance: {clause['cross_score']:.3f})*"
                            )
                            st.caption(clause["text"][:400] + ("…" if len(clause["text"]) > 400 else ""))
                            st.divider()

                st.caption(adverse["citation"])

            # Pre-populate the Copilot tab with this case
            st.session_state["copilot_context"] = dict(
                application=application_ctx, prob=prob, decision=decision,
                reasons=adverse.get("reasons", []),
            )
            if decision == "Reject":
                st.info(
                    "Switch to the **Regulatory Copilot** tab to ask compliance "
                    "questions about this rejection with full cited answers."
                )

        # Log + auto-queue borderline cases
        pred_id = log_prediction(
            income_lpa, age_years, exp_years,
            income_n, age_n, exp_n,
            prob, ci_lower, ci_upper,
            decision, confidence, page="Chatbot",
        )
        if decision == "Manual Review" or confidence == "Low (borderline)":
            hitl_enqueue(
                pred_id, income_lpa, age_years, exp_years, prob, ci_lower, ci_upper,
                reason="Borderline case from Chatbot",
            )
            st.info("This case has been added to the analyst review queue.")

    # ── Feedback ────────────────────────────────────────────────────────────
    if "chatbot_pred" in st.session_state:
        st.markdown("---")
        fb_c1, fb_c2 = st.columns(2)
        with fb_c1:
            feedback = st.radio(
                "Was this prediction correct?",
                ["correct", "incorrect", "unsure"],
                horizontal=True, key="chatbot_fb_radio",
            )
        with fb_c2:
            corrected = st.selectbox(
                "Correct label:",
                ["", "Should be Approve", "Should be Reject", "Should be Review"],
                key="chatbot_fb_select",
            )
        notes = st.text_input("Notes:", placeholder="Optional context", key="chatbot_fb_notes")
        if st.button("Submit Feedback", key="chatbot_fb_submit"):
            log_feedback("chatbot", feedback, corrected, notes)
            st.success("Feedback recorded.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Regulatory Copilot
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Regulatory RAG Copilot")
    st.caption(
        "Ask compliance questions about credit decisions. Answers are grounded in "
        "retrieved clauses from the RBI IT Framework, DPDP Act 2023, ECOA Regulation B, "
        "and FCRA — with inline citations to the specific clause retrieved."
    )

    # ── Load RAG agent ───────────────────────────────────────────────────────
    retriever, rag_ok = _load_rag_agent()

    if rag_ok is not True:
        st.error(
            f"RAG dependencies not available: `{rag_ok}`\n\n"
            "Install with:\n"
            "```\npip install chromadb sentence-transformers langchain-core langgraph\n```\n"
            "Then restart the app."
        )
        st.stop()

    # ── Context banner (when a prediction has been made) ────────────────────
    copilot_ctx = st.session_state.get("copilot_context", {})
    if copilot_ctx:
        app  = copilot_ctx.get("application", {})
        prob = copilot_ctx.get("prob", 0.0)
        st.info(
            f"**Active application context** — "
            f"Income: {app.get('income_lpa', '?')} LPA | "
            f"Age: {app.get('age_years', '?')} yrs | "
            f"Experience: {app.get('exp_years', '?')} yrs | "
            f"Risk score: {prob:.2%} | "
            f"Decision: {copilot_ctx.get('decision', '?')}   "
            f"*(questions will use this as application context)*"
        )

    # ── Suggested questions ──────────────────────────────────────────────────
    with st.expander("💡 Suggested questions", expanded=False):
        cols_q = st.columns(2)
        for i, q in enumerate(_STARTER_QUESTIONS):
            if cols_q[i % 2].button(q, key=f"sq_{i}", use_container_width=True):
                st.session_state["copilot_input"] = q

    # ── Chat history init ────────────────────────────────────────────────────
    if "copilot_messages" not in st.session_state:
        st.session_state["copilot_messages"] = []

    # ── Render chat history ──────────────────────────────────────────────────
    for msg in st.session_state["copilot_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("clauses"):
                with st.expander("📚 Retrieved regulatory clauses", expanded=False):
                    for c in msg["clauses"]:
                        st.markdown(
                            f"**[{c['rank']}] {c['citation']}** "
                            f"*(relevance: {c['cross_score']:.3f})*"
                        )
                        st.caption(c["text"][:500] + ("…" if len(c["text"]) > 500 else ""))
                        st.divider()

    # ── Chat input ───────────────────────────────────────────────────────────
    prefill = st.session_state.pop("copilot_input", "")
    user_query = st.chat_input(
        placeholder="e.g. Is this decision compliant with ECOA? What does DPDP §12 require?",
        key="copilot_chat_input",
    ) or prefill

    if user_query:
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state["copilot_messages"].append(
            {"role": "user", "content": user_query}
        )

        # Run the RAG agent
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant regulatory clauses…"):
                try:
                    ctx     = st.session_state.get("copilot_context", {})
                    result  = _run_rag_query(
                        query=user_query,
                        application=ctx.get("application"),
                        denial_reasons=ctx.get("reasons"),
                        risk_score=ctx.get("prob", 0.0),
                    )

                    response  = result.get("response", "")
                    intent    = result.get("intent", "general_query")
                    clauses   = result.get("retrieved_clauses", [])
                    citations = result.get("citations", [])
                    error     = result.get("error", "")

                    # ── Display response ─────────────────────────────────
                    if intent == "adverse_action" and result.get("notice"):
                        st.markdown(result["notice"])
                    else:
                        st.markdown(response)

                    # ── Citations pill row ───────────────────────────────
                    if citations:
                        st.markdown(
                            " &nbsp;·&nbsp; ".join(
                                f"`{c}`" for c in citations[:5]
                            ),
                        )

                    # ── Retrieved clauses expander ───────────────────────
                    if clauses:
                        with st.expander(
                            f"📚 {len(clauses)} regulatory clause(s) retrieved",
                            expanded=False,
                        ):
                            for c in clauses:
                                col_a, col_b = st.columns([3, 1])
                                with col_a:
                                    st.markdown(f"**{c['citation']}**")
                                    st.caption(
                                        c["text"][:500]
                                        + ("…" if len(c["text"]) > 500 else "")
                                    )
                                with col_b:
                                    st.metric(
                                        "Relevance",
                                        f"{c['cross_score']:.3f}",
                                        help="Cross-encoder score (higher = more relevant)",
                                    )
                                st.divider()

                    if error:
                        st.warning(f"⚠️ {error}")

                    # ── Save to history ──────────────────────────────────
                    assistant_content = (
                        result.get("notice") or response
                        if intent == "adverse_action"
                        else response
                    )
                    st.session_state["copilot_messages"].append({
                        "role":    "assistant",
                        "content": assistant_content,
                        "clauses": clauses,
                    })

                except Exception as exc:
                    err_msg = (
                        f"RAG agent error: {exc}\n\n"
                        "Ensure the regulatory store is initialised:\n"
                        "```\npython -m src.ingest_regulations\n```"
                    )
                    st.error(err_msg)
                    st.session_state["copilot_messages"].append(
                        {"role": "assistant", "content": err_msg, "clauses": []}
                    )

    # ── Sidebar controls ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🏛 Regulatory Copilot")
        st.markdown(
            "**Knowledge base:**\n"
            "- RBI IT Framework (2023)\n"
            "- DPDP Act 2023\n"
            "- ECOA Regulation B\n"
            "- FCRA\n\n"
            "**Retrieval:** ChromaDB + cross-encoder reranker\n\n"
            "**LLM:** GPT-4o-mini › Groq Llama-3 › Template fallback"
        )
        st.divider()

        if st.button("🗑 Clear chat history", use_container_width=True):
            st.session_state["copilot_messages"] = []
            st.rerun()

        if st.button("🔄 Clear application context", use_container_width=True):
            st.session_state.pop("copilot_context", None)
            st.rerun()

        st.divider()
        st.markdown("**Store stats**")
        try:
            from src.rag_ingest import get_collection_stats
            stats = get_collection_stats()
            st.metric("Total chunks", stats.get("total_chunks", 0))
            if stats.get("sources"):
                for src, cnt in list(stats["sources"].items())[:4]:
                    short = src.split("(")[0].strip()[:30]
                    st.caption(f"{cnt} — {short}")
        except Exception:
            st.caption("Stats unavailable")

        st.divider()
        st.markdown(
            "**Add more regulations:**\n"
            "Drop PDFs into `data/regulatory_docs/` then run:\n"
            "```\npython -m src.ingest_regulations\n```"
        )
