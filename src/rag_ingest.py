"""
Regulatory RAG Ingest Pipeline.

Parses regulatory PDFs (RBI IT Framework, DPDP Act 2023, ECOA/FCRA),
chunks them with overlap, embeds via sentence-transformers, and persists
to a ChromaDB collection at data/regulatory_chroma/.

Falls back to a built-in seed knowledge base when no PDFs are present,
so the RAG copilot works out-of-the-box without requiring PDF downloads.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
CHROMA_DIR = BASE_DIR / "data" / "regulatory_chroma"
DOCS_DIR = BASE_DIR / "data" / "regulatory_docs"

COLLECTION_NAME = "regulatory_docs"
CHUNK_SIZE = 800        # characters (≈ 200 tokens for MiniLM)
CHUNK_OVERLAP = 120     # characters

# ── Seed Knowledge Base ───────────────────────────────────────────────────────
# Compiled from publicly available regulatory summaries.
# Structured as: {clause_id, source, text}

SEED_CLAUSES: list[dict[str, str]] = [
    # ── RBI IT Framework 2023 ──────────────────────────────────────────────
    {
        "clause_id": "RBI-IT-Sec2.1",
        "source": "RBI Master Direction on IT Framework (2023)",
        "page": "Section 2.1",
        "text": (
            "Regulated Entities (REs) shall have a Board-approved IT policy covering IT governance, "
            "risk management, infrastructure, data management, and cyber security. The policy shall "
            "be reviewed at least annually and after major incidents. Senior management is responsible "
            "for implementation and monitoring of the IT policy."
        ),
    },
    {
        "clause_id": "RBI-IT-Sec3.1",
        "source": "RBI Master Direction on IT Framework (2023)",
        "page": "Section 3.1",
        "text": (
            "REs shall classify data based on sensitivity and business criticality. Personal and "
            "financial data of customers shall be classified as 'Confidential' or above. Access to "
            "classified data shall be restricted on a need-to-know basis with appropriate controls. "
            "Data classification policy shall be reviewed annually."
        ),
    },
    {
        "clause_id": "RBI-IT-Sec4.2",
        "source": "RBI Master Direction on IT Framework (2023)",
        "page": "Section 4.2",
        "text": (
            "REs shall implement encryption for data at rest and data in transit. Sensitive customer "
            "data, including credit information, shall be encrypted using AES-256 or equivalent. "
            "Encryption keys shall be managed through a formal key management process with regular "
            "rotation, secure storage, and access controls."
        ),
    },
    {
        "clause_id": "RBI-IT-Sec5.1",
        "source": "RBI Master Direction on IT Framework (2023)",
        "page": "Section 5.1",
        "text": (
            "REs shall maintain comprehensive audit trails for all critical system activities including "
            "user access, data modifications, system configuration changes, and credit decisions. "
            "Audit logs shall be tamper-proof, retained for a minimum of 5 years, and regularly "
            "reviewed. Hash-based integrity verification is recommended."
        ),
    },
    {
        "clause_id": "RBI-IT-Sec6.3",
        "source": "RBI Master Direction on IT Framework (2023)",
        "page": "Section 6.3",
        "text": (
            "Algorithmic models used for credit decisions must be documented, validated, and subject "
            "to model risk management. REs shall conduct periodic model validation including back-testing, "
            "stress testing, and sensitivity analysis. Model changes shall follow a formal change "
            "management process with documented approval."
        ),
    },
    {
        "clause_id": "RBI-IT-Sec7.1",
        "source": "RBI Master Direction on IT Framework (2023)",
        "page": "Section 7.1",
        "text": (
            "REs shall implement role-based access control (RBAC) for all systems. Privileged access "
            "shall be managed through a Privileged Access Management (PAM) solution. Access rights "
            "shall be reviewed quarterly. Dormant accounts shall be disabled after 60 days of inactivity. "
            "Multi-factor authentication (MFA) is mandatory for privileged and remote access."
        ),
    },
    {
        "clause_id": "RBI-IT-Sec8.2",
        "source": "RBI Master Direction on IT Framework (2023)",
        "page": "Section 8.2",
        "text": (
            "REs shall conduct annual Vulnerability Assessment and Penetration Testing (VAPT) on "
            "critical systems. External-facing systems shall undergo VAPT before deployment and after "
            "significant changes. Identified vulnerabilities shall be remediated within defined SLAs "
            "based on severity: Critical (7 days), High (30 days), Medium (90 days)."
        ),
    },
    {
        "clause_id": "RBI-IT-Sec9.1",
        "source": "RBI Master Direction on IT Framework (2023)",
        "page": "Section 9.1",
        "text": (
            "REs shall develop and maintain a Business Continuity Plan (BCP) and Disaster Recovery (DR) "
            "plan. Recovery Time Objective (RTO) for critical systems shall not exceed 4 hours. "
            "Recovery Point Objective (RPO) shall not exceed 2 hours. BCP/DR plans shall be tested "
            "at least annually through live drills."
        ),
    },
    {
        "clause_id": "RBI-IT-Sec10.4",
        "source": "RBI Master Direction on IT Framework (2023)",
        "page": "Section 10.4",
        "text": (
            "Third-party service providers with access to customer data or critical systems shall be "
            "subject to due diligence before onboarding and periodic risk assessments. Contracts shall "
            "include data security obligations, right-to-audit clauses, and breach notification "
            "requirements. Sub-contracting of critical functions requires prior approval from the RE."
        ),
    },
    # ── DPDP Act 2023 ─────────────────────────────────────────────────────
    {
        "clause_id": "DPDP-Sec4",
        "source": "Digital Personal Data Protection Act 2023 (India)",
        "page": "Section 4",
        "text": (
            "Personal data may be processed only for a lawful purpose for which the Data Principal "
            "has given consent, or for certain legitimate uses specified in the Act. Consent must be "
            "free, specific, informed, unconditional, and unambiguous, expressed through a clear "
            "affirmative action. Consent shall not be obtained through deception or coercion."
        ),
    },
    {
        "clause_id": "DPDP-Sec6",
        "source": "Digital Personal Data Protection Act 2023 (India)",
        "page": "Section 6",
        "text": (
            "The Data Fiduciary shall provide the Data Principal with a clear and plain language "
            "notice before obtaining consent. The notice shall specify: the personal data to be "
            "processed, the purpose of processing, and the manner in which the Data Principal may "
            "exercise their rights under the Act including right to withdraw consent."
        ),
    },
    {
        "clause_id": "DPDP-Sec8",
        "source": "Digital Personal Data Protection Act 2023 (India)",
        "page": "Section 8",
        "text": (
            "Every Data Fiduciary shall ensure completeness, accuracy, and consistency of personal "
            "data. The Data Fiduciary shall implement appropriate technical and organisational measures "
            "to ensure data quality. Inaccurate data that could adversely impact the Data Principal "
            "shall be corrected promptly upon notice."
        ),
    },
    {
        "clause_id": "DPDP-Sec9",
        "source": "Digital Personal Data Protection Act 2023 (India)",
        "page": "Section 9",
        "text": (
            "The Data Fiduciary shall not retain personal data beyond the period necessary to fulfil "
            "the purpose for which it was collected. Upon achievement of the purpose or withdrawal of "
            "consent, personal data shall be erased within the prescribed period. A retention policy "
            "with defined data lifecycle stages must be documented and enforced."
        ),
    },
    {
        "clause_id": "DPDP-Sec11",
        "source": "Digital Personal Data Protection Act 2023 (India)",
        "page": "Section 11",
        "text": (
            "The Data Principal shall have the right to: (a) obtain a summary of personal data "
            "processed and processing activities; (b) know the identities of all Data Fiduciaries and "
            "processors with whom personal data has been shared; (c) access other information as may "
            "be prescribed. Data Fiduciaries must respond to access requests within 30 days."
        ),
    },
    {
        "clause_id": "DPDP-Sec12",
        "source": "Digital Personal Data Protection Act 2023 (India)",
        "page": "Section 12",
        "text": (
            "The Data Principal has the right to correction and erasure of personal data. The Data "
            "Fiduciary shall correct inaccurate or misleading data, complete incomplete data, and "
            "update outdated data. The right to erasure applies unless retention is required by law "
            "or for legal proceedings. Erasure requests must be fulfilled within 30 days."
        ),
    },
    {
        "clause_id": "DPDP-Sec14",
        "source": "Digital Personal Data Protection Act 2023 (India)",
        "page": "Section 14",
        "text": (
            "The Data Principal shall have the right to grieve. Every Data Fiduciary shall establish "
            "an effective grievance redressal mechanism. Complaints shall be acknowledged within 72 "
            "hours and resolved within 30 days. The Data Principal may approach the Data Protection "
            "Board if not satisfied with the resolution."
        ),
    },
    {
        "clause_id": "DPDP-Sec16",
        "source": "Digital Personal Data Protection Act 2023 (India)",
        "page": "Section 16",
        "text": (
            "Every Data Fiduciary and Data Processor shall implement appropriate technical and "
            "organisational measures to ensure effective observance of the Act. These shall include "
            "security safeguards to prevent personal data breach, including encryption, access controls, "
            "and pseudonymisation. In case of a data breach, the Board and affected Data Principals "
            "shall be intimated in the prescribed manner."
        ),
    },
    # ── ECOA / Regulation B ────────────────────────────────────────────────
    {
        "clause_id": "ECOA-RegB-9a",
        "source": "Equal Credit Opportunity Act – Regulation B (12 CFR 1002)",
        "page": "§1002.9(a)",
        "text": (
            "A creditor shall notify an applicant of action taken on a credit application within 30 "
            "days after receiving a completed application. For adverse action, the notice must be in "
            "writing and contain: (1) a statement of the action taken; (2) the name and address of "
            "the creditor; (3) a statement of the applicant's right to a statement of specific reasons "
            "within 60 days; and (4) the name and address of the person or office that will provide "
            "the statement of reasons."
        ),
    },
    {
        "clause_id": "ECOA-RegB-9b",
        "source": "Equal Credit Opportunity Act – Regulation B (12 CFR 1002)",
        "page": "§1002.9(b)",
        "text": (
            "The statement of reasons for adverse action must be specific and indicate the principal "
            "reasons for the action. Reasons such as 'our internal standards' or 'your application "
            "did not meet our criteria' are insufficient. The creditor must disclose the actual reasons "
            "such as 'insufficient income', 'excessive obligations', 'limited credit history', or "
            "'unable to verify income'. When credit scoring is used, the four most significant factors "
            "must be disclosed."
        ),
    },
    {
        "clause_id": "ECOA-RegB-6",
        "source": "Equal Credit Opportunity Act – Regulation B (12 CFR 1002)",
        "page": "§1002.6",
        "text": (
            "A creditor shall not consider race, color, religion, national origin, sex, marital "
            "status, or age (provided the applicant has the capacity to contract) in any aspect of a "
            "credit transaction. A creditor shall not use a prohibited basis as a proxy for a "
            "non-prohibited factor. Credit scoring systems must be empirically derived and "
            "statistically sound, and must not assign a negative factor to age."
        ),
    },
    {
        "clause_id": "ECOA-RegB-5b",
        "source": "Equal Credit Opportunity Act – Regulation B (12 CFR 1002)",
        "page": "§1002.5(b)",
        "text": (
            "A creditor shall not request information about an applicant's race, color, religion, "
            "national origin, or sex except as required by law. Exceptions apply for home mortgage "
            "applications where such data is required for HMDA reporting. The creditor must notify "
            "the applicant that disclosure is voluntary for monitoring purposes."
        ),
    },
    # ── FCRA ───────────────────────────────────────────────────────────────
    {
        "clause_id": "FCRA-Sec615a",
        "source": "Fair Credit Reporting Act (15 U.S.C. § 1681)",
        "page": "§615(a)",
        "text": (
            "If any person takes any adverse action with respect to any consumer that is based in "
            "whole or in part on any information contained in a consumer report, the person shall "
            "provide written notice to the consumer containing: (1) the name, address, and phone "
            "number of the consumer reporting agency; (2) a statement that the agency did not make "
            "the adverse decision; (3) notice of the consumer's right to obtain a free copy of the "
            "report within 60 days; and (4) the consumer's right to dispute accuracy of the report."
        ),
    },
    {
        "clause_id": "FCRA-Sec615b",
        "source": "Fair Credit Reporting Act (15 U.S.C. § 1681)",
        "page": "§615(b)",
        "text": (
            "Whenever a user of consumer reports takes an adverse action based on information from "
            "an investigative consumer report, the consumer must be informed of their right to request "
            "a complete and accurate disclosure of the nature and scope of the investigation, within "
            "5 days of the request. The user must comply within a reasonable time."
        ),
    },
    {
        "clause_id": "FCRA-Sec611",
        "source": "Fair Credit Reporting Act (15 U.S.C. § 1681)",
        "page": "§611",
        "text": (
            "If a consumer disputes the completeness or accuracy of any item of information in a "
            "consumer report, the consumer reporting agency shall conduct a free reinvestigation "
            "within 30 days (extendable to 45 days). If the disputed information cannot be verified, "
            "it shall be deleted or corrected. The consumer must be notified of the results within "
            "5 business days of completion."
        ),
    },
    {
        "clause_id": "FCRA-Sec604",
        "source": "Fair Credit Reporting Act (15 U.S.C. § 1681)",
        "page": "§604",
        "text": (
            "A consumer reporting agency may furnish a consumer report only under defined permissible "
            "purposes including: written instructions of the consumer; credit transactions initiated "
            "by the consumer; employment purposes with written authorization; insurance underwriting; "
            "and court orders. Use of a consumer report for any other purpose is a violation of the "
            "FCRA and may result in civil liability."
        ),
    },
]


# ── Chunker ───────────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level chunks, breaking on sentence boundaries."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Try to break at a sentence boundary within the last 20% of the chunk
        if end < len(text):
            search_from = max(start, end - chunk_size // 5)
            m = None
            for m in re.finditer(r'[.!?]\s+', text[search_from:end]):
                pass  # find last match
            if m:
                end = search_from + m.end()
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if len(c) > 50]


def _pdf_to_text(pdf_path: Path) -> list[dict[str, Any]]:
    """Extract text from a PDF, returning list of {page, text} dicts."""
    try:
        import pypdf
        reader = pypdf.PdfReader(str(pdf_path))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": i + 1, "text": text})
        return pages
    except ImportError:
        logger.warning("pypdf not installed — skipping PDF %s", pdf_path.name)
        return []
    except Exception as exc:
        logger.error("Failed to read %s: %s", pdf_path.name, exc)
        return []


def _detect_clause_id(text: str, source_hint: str) -> str:
    """Auto-detect clause/section IDs from heading patterns in text."""
    patterns = [
        r'\bSection\s+(\d+[\.\d]*)',
        r'\b§\s*(\d+[\.\d]*[a-z]?)',
        r'\bArt(?:icle)?\.\s*(\d+)',
        r'\bClause\s+(\d+[\.\d]*)',
        r'\bRule\s+(\d+[\.\d]*)',
        r'\bParagraph\s+(\d+[\.\d]*)',
        r'\bPara\s+(\d+[\.\d]*)',
    ]
    for pat in patterns:
        m = re.search(pat, text[:200], re.IGNORECASE)
        if m:
            prefix = source_hint[:4].upper().replace(" ", "")
            return f"{prefix}-{m.group(1)}"
    return ""


def pdf_to_chunks(pdf_path: Path, source_name: str | None = None) -> list[dict[str, Any]]:
    """
    Parse a PDF and return a list of chunk dicts ready for ChromaDB insertion:
    {id, document, source, page, clause_id}
    """
    name = source_name or pdf_path.stem.replace("_", " ").title()
    pages = _pdf_to_text(pdf_path)
    chunks: list[dict[str, Any]] = []

    for page_info in pages:
        for chunk_text in _chunk_text(page_info["text"]):
            clause_id = _detect_clause_id(chunk_text, name)
            doc_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
            chunks.append({
                "id": f"{pdf_path.stem}-p{page_info['page']}-{doc_hash}",
                "document": chunk_text,
                "source": name,
                "page": str(page_info["page"]),
                "clause_id": clause_id,
            })

    logger.info("Parsed %s → %d chunks", pdf_path.name, len(chunks))
    return chunks


# ── ChromaDB Store ────────────────────────────────────────────────────────────

def _get_client():
    """Return a persistent ChromaDB client."""
    import chromadb
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _get_embedding_fn():
    """Return ChromaDB-compatible SentenceTransformer embedding function."""
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    return SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def get_or_create_collection(reset: bool = False):
    """Get or create the regulatory_docs collection."""
    client = _get_client()
    ef = _get_embedding_fn()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info("Deleted existing collection '%s'", COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def _batch_upsert(collection, chunks: list[dict[str, Any]], batch_size: int = 100) -> int:
    """Upsert chunks in batches; returns count of newly added chunks."""
    existing_ids: set[str] = set()
    try:
        all_ids = collection.get(include=[])["ids"]
        existing_ids = set(all_ids)
    except Exception:
        pass

    new_chunks = [c for c in chunks if c["id"] not in existing_ids]
    if not new_chunks:
        return 0

    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i : i + batch_size]
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["document"] for c in batch],
            metadatas=[
                {"source": c["source"], "page": c["page"], "clause_id": c["clause_id"]}
                for c in batch
            ],
        )
    return len(new_chunks)


def ingest_seed_knowledge(collection, reset: bool = False) -> int:
    """Load the built-in seed regulatory knowledge base into ChromaDB."""
    chunks = []
    for clause in SEED_CLAUSES:
        doc_hash = hashlib.md5(clause["text"].encode()).hexdigest()[:8]
        chunks.append({
            "id": f"seed-{clause['clause_id']}-{doc_hash}",
            "document": clause["text"],
            "source": clause["source"],
            "page": clause["page"],
            "clause_id": clause["clause_id"],
        })
        # Also chunk long clauses
        if len(clause["text"]) > CHUNK_SIZE:
            for sub in _chunk_text(clause["text"]):
                sub_hash = hashlib.md5(sub.encode()).hexdigest()[:8]
                chunks.append({
                    "id": f"seed-{clause['clause_id']}-sub-{sub_hash}",
                    "document": sub,
                    "source": clause["source"],
                    "page": clause["page"],
                    "clause_id": clause["clause_id"],
                })

    added = _batch_upsert(collection, chunks)
    logger.info("Seed knowledge base: %d new chunks added", added)
    return added


def ingest_pdf(pdf_path: Path, collection, source_name: str | None = None) -> int:
    """Parse a single PDF and upsert into ChromaDB. Returns count of new chunks."""
    chunks = pdf_to_chunks(pdf_path, source_name)
    if not chunks:
        return 0
    return _batch_upsert(collection, chunks)


def ingest_all_pdfs(docs_dir: Path | None = None, reset: bool = False) -> dict[str, int]:
    """
    Ingest all PDFs from docs_dir plus the seed knowledge base.
    Returns {filename: chunk_count} summary.
    """
    docs_dir = docs_dir or DOCS_DIR
    collection = get_or_create_collection(reset=reset)

    results: dict[str, int] = {}

    # Always load seed knowledge
    results["[seed]"] = ingest_seed_knowledge(collection, reset=reset)

    # Ingest any PDFs present
    pdf_files = list(docs_dir.glob("*.pdf"))
    if pdf_files:
        logger.info("Found %d PDF(s) to ingest", len(pdf_files))
        for pdf_path in pdf_files:
            count = ingest_pdf(pdf_path, collection)
            results[pdf_path.name] = count
    else:
        logger.info("No PDFs found in %s — using seed knowledge only", docs_dir)

    total = collection.count()
    logger.info("Total chunks in store: %d", total)
    results["__total__"] = total
    return results


def get_collection_stats() -> dict[str, Any]:
    """Return stats about the current ChromaDB collection."""
    try:
        collection = get_or_create_collection()
        count = collection.count()
        # Sample metadata for source breakdown
        if count > 0:
            sample = collection.get(limit=min(count, 500), include=["metadatas"])
            sources: dict[str, int] = {}
            for meta in sample["metadatas"]:
                src = meta.get("source", "unknown")
                sources[src] = sources.get(src, 0) + 1
        else:
            sources = {}
        return {"total_chunks": count, "sources": sources, "chroma_dir": str(CHROMA_DIR)}
    except Exception as e:
        return {"error": str(e), "total_chunks": 0, "sources": {}}
