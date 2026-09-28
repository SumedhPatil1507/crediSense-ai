"""
RAG Retriever with Cross-Encoder Reranker.

1. Bi-encoder retrieval  — ChromaDB ANN search (top-k candidates)
2. Cross-encoder rerank  — ms-marco-MiniLM-L-6-v2 scores each candidate
3. Returns top-n ranked results with metadata for citation generation

Design:
- Graceful fallback if sentence-transformers or cross-encoder not installed
- Thread-safe singleton for the cross-encoder model
- All results carry {text, source, page, clause_id, score} for inline citation
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class RetrievedClause:
    """A single retrieved and reranked regulatory clause."""
    text: str
    source: str
    page: str
    clause_id: str
    bi_score: float       # cosine similarity from ChromaDB (lower = more similar for distance)
    cross_score: float    # cross-encoder relevance score (higher = more relevant)
    rank: int             # final rank (1 = most relevant)

    @property
    def citation(self) -> str:
        """Short citation string for inline use."""
        parts = [self.source]
        if self.clause_id:
            parts.append(self.clause_id)
        elif self.page:
            parts.append(f"p.{self.page}")
        return " · ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "clause_id": self.clause_id,
            "citation": self.citation,
            "cross_score": round(self.cross_score, 4),
            "rank": self.rank,
        }


# ── Cross-Encoder Singleton ───────────────────────────────────────────────────

_CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_ce_model = None
_ce_lock = threading.Lock()


def _get_cross_encoder():
    """Load cross-encoder lazily (thread-safe)."""
    global _ce_model
    if _ce_model is None:
        with _ce_lock:
            if _ce_model is None:
                try:
                    from sentence_transformers import CrossEncoder
                    _ce_model = CrossEncoder(_CE_MODEL_NAME, max_length=512)
                    logger.info("Cross-encoder loaded: %s", _CE_MODEL_NAME)
                except ImportError:
                    logger.warning(
                        "sentence-transformers not installed; cross-encoder reranking disabled"
                    )
                    _ce_model = None
                except Exception as e:
                    logger.error("Failed to load cross-encoder: %s", e)
                    _ce_model = None
    return _ce_model


# ── Retriever ─────────────────────────────────────────────────────────────────

class RegulatoryRetriever:
    """
    Retrieves relevant regulatory clauses for a query using:
    1. ChromaDB bi-encoder ANN search (fast candidate generation)
    2. Cross-encoder reranking (precision re-scoring)
    """

    def __init__(
        self,
        n_candidates: int = 20,
        top_n: int = 5,
        min_cross_score: float = -5.0,
    ):
        """
        Args:
            n_candidates: Number of candidates to fetch from ChromaDB before reranking.
            top_n: Final number of results to return after reranking.
            min_cross_score: Minimum cross-encoder score threshold (logit scale, ms-marco model).
        """
        self.n_candidates = n_candidates
        self.top_n = top_n
        self.min_cross_score = min_cross_score
        self._collection = None

    def _get_collection(self):
        """Lazily get the ChromaDB collection."""
        if self._collection is None:
            from src.rag_ingest import get_or_create_collection, ingest_seed_knowledge
            collection = get_or_create_collection()
            # Auto-seed if empty
            if collection.count() == 0:
                logger.info("Collection empty — loading seed knowledge base")
                ingest_seed_knowledge(collection)
            self._collection = collection
        return self._collection

    def retrieve(self, query: str, filter_source: str | None = None) -> list[RetrievedClause]:
        """
        Retrieve and rerank the most relevant regulatory clauses for a query.

        Args:
            query: Natural language query.
            filter_source: Optional source document to restrict results to.

        Returns:
            List of RetrievedClause ordered by cross-encoder score descending.
        """
        collection = self._get_collection()
        if collection.count() == 0:
            logger.warning("No documents in collection — returning empty results")
            return []

        # ── Step 1: Bi-encoder ANN retrieval ─────────────────────────────
        where_filter = None
        if filter_source:
            where_filter = {"source": {"$eq": filter_source}}

        try:
            results = collection.query(
                query_texts=[query],
                n_results=min(self.n_candidates, collection.count()),
                include=["documents", "metadatas", "distances"],
                where=where_filter,
            )
        except Exception as e:
            logger.error("ChromaDB query failed: %s", e)
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        if not docs:
            return []

        # ── Step 2: Cross-encoder reranking ───────────────────────────────
        ce_model = _get_cross_encoder()
        if ce_model is not None:
            pairs = [[query, doc] for doc in docs]
            try:
                scores = ce_model.predict(pairs).tolist()
            except Exception as e:
                logger.error("Cross-encoder scoring failed: %s", e)
                # Fall back to bi-encoder distance (invert distance for sorting)
                scores = [1.0 - d for d in distances]
        else:
            # Fallback: use inverted cosine distance as proxy
            scores = [1.0 - d for d in distances]

        # ── Step 3: Sort by cross-encoder score ───────────────────────────
        ranked = sorted(
            zip(docs, metas, distances, scores),
            key=lambda x: x[3],
            reverse=True,
        )

        # Filter by minimum score and take top_n
        clauses: list[RetrievedClause] = []
        for rank, (doc, meta, dist, score) in enumerate(ranked[:self.top_n], start=1):
            if score < self.min_cross_score:
                continue
            clauses.append(
                RetrievedClause(
                    text=doc,
                    source=meta.get("source", "Unknown"),
                    page=meta.get("page", ""),
                    clause_id=meta.get("clause_id", ""),
                    bi_score=float(1.0 - dist),
                    cross_score=float(score),
                    rank=rank,
                )
            )

        logger.debug(
            "Query '%s…': %d candidates → %d after rerank",
            query[:60], len(docs), len(clauses)
        )
        return clauses

    def retrieve_for_adverse_action(
        self, reasons: list[str], regulation_scope: str | None = None
    ) -> list[RetrievedClause]:
        """
        Retrieve clauses specifically relevant to an adverse action notice.
        Constructs a targeted query from the denial reasons.

        Args:
            reasons: List of denial reason strings from generate_adverse_action().
            regulation_scope: Optional filter e.g. 'ECOA', 'RBI', 'DPDP'.

        Returns:
            Ranked list of relevant regulatory clauses.
        """
        query = (
            "adverse action notice credit denial requirements obligations: "
            + "; ".join(reasons)
        )
        return self.retrieve(query)

    def format_context_block(
        self, clauses: list[RetrievedClause], max_chars: int = 4000
    ) -> str:
        """
        Format retrieved clauses into a numbered context block for the LLM prompt.
        Truncates to max_chars to stay within token budget.
        """
        lines = []
        total = 0
        for c in clauses:
            block = (
                f"[{c.rank}] {c.citation}\n"
                f"{c.text}\n"
            )
            if total + len(block) > max_chars:
                break
            lines.append(block)
            total += len(block)
        return "\n---\n".join(lines)


# ── Module-level singleton ────────────────────────────────────────────────────

_retriever: RegulatoryRetriever | None = None
_retriever_lock = threading.Lock()


def get_retriever(n_candidates: int = 20, top_n: int = 5) -> RegulatoryRetriever:
    """Return the module-level retriever singleton (lazy init)."""
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = RegulatoryRetriever(
                    n_candidates=n_candidates, top_n=top_n
                )
    return _retriever
