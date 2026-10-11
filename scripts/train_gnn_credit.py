"""Train the CrediSense heterogeneous R-GCN and export node scores to Neo4j."""
from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gnn_risk import (  # noqa: E402
    DEFAULT_FEATURE_COLUMNS,
    build_credit_graph,
    score_applicants,
    train_model,
    write_scores_to_neo4j,
)

LOGGER = logging.getLogger("train_gnn_credit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a two-layer heterogeneous R-GCN and persist applicant embeddings to Neo4j."
    )
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "loan_cleaned.csv",
                        help="CSV with applicant features and a binary Risk_Flag label")
    parser.add_argument("--label-column", default="Risk_Flag")
    parser.add_argument("--syndicate-label-column", default="Syndicate_Risk",
                        help="Optional [0,1] auxiliary label; otherwise use linked-applicant observed default rate")
    parser.add_argument("--feature-columns", nargs="+", default=list(DEFAULT_FEATURE_COLUMNS),
                        help="Numeric applicant feature columns available in the CSV")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "gnn_credit.pt",
                        help="Checkpoint output path")
    parser.add_argument("--skip-neo4j-write", action="store_true",
                        help="Train and save the checkpoint without exporting to Neo4j")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    frame = pd.read_csv(args.input)
    auxiliary_label = args.syndicate_label_column if args.syndicate_label_column in frame.columns else None
    graph = build_credit_graph(
        frame,
        label_column=args.label_column,
        feature_columns=args.feature_columns,
        syndicate_label_column=auxiliary_label,
    )
    model, metrics = train_model(
        graph,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    rows = score_applicants(model, graph)
    model_version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save({
        "model_state_dict": model.state_dict(),
        "node_types": list(graph.data.node_types),
        "edge_types": [list(edge) for edge in graph.data.edge_types],
        "feature_columns": graph.feature_columns,
        "input_dim": len(graph.feature_columns),
        "hidden_dim": args.hidden_dim,
        "label_column": args.label_column,
        "syndicate_label_column": auxiliary_label,
        "syndicate_target_method": "explicit_label" if auxiliary_label else "linked_neighbor_default_rate",
        "model_version": model_version,
        "training_metrics": metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }, temporary)
    os.replace(temporary, args.output)
    LOGGER.info("Trained %d applicants; loss %.5f; checkpoint: %s",
                len(rows), metrics["training_loss"], args.output)
    if args.skip_neo4j_write:
        LOGGER.info("Neo4j export skipped by request")
    else:
        written = write_scores_to_neo4j(rows, model_version=model_version)
        LOGGER.info("Wrote %d applicant embedding and risk records to Neo4j", written)


if __name__ == "__main__":
    main()
