"""
CLI script to ingest regulatory PDFs into the ChromaDB vector store.

Usage:
    python -m src.ingest_regulations                              # ingest all PDFs + seed
    python -m src.ingest_regulations --file path/to/doc.pdf       # single file
    python -m src.ingest_regulations --reset                      # wipe + re-ingest
    python -m src.ingest_regulations --stats                      # show collection stats
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on the path when run as a module
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _check_deps() -> bool:
    """Verify required packages are importable before starting."""
    missing = []
    for pkg in ["chromadb", "sentence_transformers"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        logger.error(
            "Missing dependencies: %s\n"
            "Install with: pip install %s",
            ", ".join(missing),
            " ".join(missing),
        )
        return False
    return True


def cmd_stats() -> None:
    """Print current ChromaDB collection statistics."""
    from src.rag_ingest import get_collection_stats
    stats = get_collection_stats()

    if "error" in stats:
        print(f"Error: {stats['error']}")
        return

    print(f"\n{'='*55}")
    print(f"  ChromaDB Collection: regulatory_docs")
    print(f"  Store path: {stats['chroma_dir']}")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"{'─'*55}")
    if stats["sources"]:
        print("  Chunks by source:")
        for src, cnt in sorted(stats["sources"].items(), key=lambda x: -x[1]):
            print(f"    {cnt:4d}  {src}")
    else:
        print("  Collection is empty.")
    print(f"{'='*55}\n")


def cmd_ingest(
    pdf_path: Path | None = None,
    reset: bool = False,
) -> None:
    """Run the full ingest pipeline."""
    from src.rag_ingest import (
        ingest_all_pdfs,
        ingest_pdf,
        get_or_create_collection,
        ingest_seed_knowledge,
        DOCS_DIR,
    )

    if pdf_path:
        if not pdf_path.exists():
            logger.error("File not found: %s", pdf_path)
            sys.exit(1)
        logger.info("Ingesting single file: %s", pdf_path.name)
        collection = get_or_create_collection(reset=reset)
        # Always ensure seed is loaded
        seed_count = ingest_seed_knowledge(collection)
        logger.info("Seed clauses: %d new chunks", seed_count)
        pdf_count = ingest_pdf(pdf_path, collection)
        logger.info("PDF '%s': %d new chunks added", pdf_path.name, pdf_count)
        total = collection.count()
        logger.info("Total chunks in store: %d", total)
    else:
        logger.info("Starting full ingest from: %s", DOCS_DIR)
        results = ingest_all_pdfs(reset=reset)
        print(f"\n{'='*55}")
        print("  Ingest Summary")
        print(f"{'─'*55}")
        for name, count in results.items():
            if name == "__total__":
                continue
            tag = "NEW" if count > 0 else "dup"
            print(f"  [{tag:3s}] {count:4d} chunks  {name}")
        print(f"{'─'*55}")
        print(f"  TOTAL chunks in store: {results.get('__total__', '?')}")
        print(f"{'='*55}\n")


def cmd_test_query(query: str) -> None:
    """Run a test retrieval query to verify the pipeline end-to-end."""
    print(f"\nTest query: \"{query}\"\n{'─'*55}")
    try:
        from src.rag_retriever import get_retriever
        retriever = get_retriever(n_candidates=10, top_n=3)
        clauses = retriever.retrieve(query)
        if not clauses:
            print("No results returned.")
            return
        for c in clauses:
            print(f"[Rank {c.rank}] Score: {c.cross_score:.3f} | {c.citation}")
            print(f"  {c.text[:200]}...")
            print()
    except Exception as e:
        logger.error("Test query failed: %s", e)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest regulatory PDFs into CrediSense ChromaDB store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.ingest_regulations                   # ingest all PDFs + seed knowledge
  python -m src.ingest_regulations --reset           # wipe and re-ingest everything
  python -m src.ingest_regulations --stats           # show collection statistics
  python -m src.ingest_regulations --file rbi.pdf    # ingest single PDF
  python -m src.ingest_regulations --test "adverse action notice requirements"
        """,
    )
    parser.add_argument(
        "--file", "-f",
        type=Path,
        default=None,
        help="Path to a single PDF to ingest (default: all PDFs in data/regulatory_docs/)",
    )
    parser.add_argument(
        "--reset", "-r",
        action="store_true",
        help="Delete the existing collection and re-ingest from scratch",
    )
    parser.add_argument(
        "--stats", "-s",
        action="store_true",
        help="Print collection statistics and exit",
    )
    parser.add_argument(
        "--test", "-t",
        type=str,
        default=None,
        metavar="QUERY",
        help="Run a test retrieval query after ingestion",
    )
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Only load the built-in seed knowledge base (no PDFs)",
    )
    args = parser.parse_args()

    if args.stats:
        cmd_stats()
        return

    if not _check_deps():
        sys.exit(1)

    if args.seed_only:
        logger.info("Loading seed knowledge base only")
        from src.rag_ingest import get_or_create_collection, ingest_seed_knowledge
        collection = get_or_create_collection(reset=args.reset)
        count = ingest_seed_knowledge(collection)
        logger.info("Seed knowledge base loaded: %d new chunks", count)
    else:
        cmd_ingest(pdf_path=args.file, reset=args.reset)

    if args.test:
        cmd_test_query(args.test)


if __name__ == "__main__":
    main()
