"""Heterogeneous R-GCN for graph-derived credit and syndicate risk signals.

The graph model is an auxiliary attribution source. Its outputs are not mixed
into the LightGBM decision probability unless a separately validated policy
explicitly does so.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, RGCNConv

LOGGER = logging.getLogger(__name__)
NODE_TYPES = ("applicant", "employer", "address", "phone_hash")
EDGE_TYPES = (
    ("applicant", "employed_at", "employer"),
    ("employer", "employs", "applicant"),
    ("applicant", "shares_address", "address"),
    ("address", "shared_by", "applicant"),
    ("applicant", "co_signed", "applicant"),
    # A phone_hash node is useful only when connected; this relation is optional
    # and accepts hashed values only.
    ("applicant", "shares_phone", "phone_hash"),
    ("phone_hash", "shared_by_phone", "applicant"),
)
DEFAULT_FEATURE_COLUMNS = (
    "Income", "Age", "Experience", "CURRENT_JOB_YRS", "CURRENT_HOUSE_YRS",
    "income_per_experience", "stability_score",
)


@dataclass
class CreditGraph:
    data: HeteroData
    applicant_ids: list[str]
    feature_columns: list[str]
    syndicate_targets: torch.Tensor


class _RelationRGCN(nn.Module):
    """Single-relation R-GCN kernel adapted to HeteroConv's typed edge inputs."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Each HeteroConv entry is one canonical relation, so its local relation
        # id is zero. RGCNConv still supplies the relation-specific transforms.
        self.conv = RGCNConv((in_channels, in_channels), out_channels,
                             num_relations=1, num_bases=1,
                             root_weight=False, bias=False)

    def forward(self, x, edge_index):
        edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
        return self.conv(x, edge_index, edge_type)


class HeteroCreditRGCN(nn.Module):
    """Two relation-aware HeteroConv layers with default and syndicate heads."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.15):
        super().__init__()
        self.node_types = NODE_TYPES
        self.input_projection = nn.ModuleDict({
            node: nn.Linear(input_dim, hidden_dim) for node in NODE_TYPES
        })
        self.conv1 = HeteroConv(
            {edge: _RelationRGCN(input_dim, hidden_dim) for edge in EDGE_TYPES}, aggr="sum"
        )
        self.conv2 = HeteroConv(
            {edge: _RelationRGCN(hidden_dim, hidden_dim) for edge in EDGE_TYPES}, aggr="sum"
        )
        self.dropout = nn.Dropout(dropout)
        self.default_head = nn.Linear(hidden_dim, 1)
        self.syndicate_head = nn.Linear(hidden_dim, 1)

    def forward(self, data: HeteroData):
        x_dict = data.x_dict
        x1_msg = self.conv1(x_dict, data.edge_index_dict)
        x1 = {
            node: F.relu(self.input_projection[node](x_dict[node]) +
                         x1_msg.get(node, torch.zeros(
                             (x_dict[node].size(0), self.default_head.in_features),
                             dtype=x_dict[node].dtype, device=x_dict[node].device)))
            for node in self.node_types
        }
        x2_msg = self.conv2(x1, data.edge_index_dict)
        embeddings = {
            node: F.relu(x1[node] + x2_msg.get(node, torch.zeros_like(x1[node])))
            for node in self.node_types
        }
        applicant_embeddings = self.dropout(embeddings["applicant"])
        return {
            "embeddings": embeddings,
            "default_logits": self.default_head(applicant_embeddings).squeeze(-1),
            "syndicate_logits": self.syndicate_head(applicant_embeddings).squeeze(-1),
        }


def _string_key(value: Any, fallback: str) -> str:
    if pd.isna(value) or not str(value).strip():
        return fallback
    return str(value).strip()


def _read_co_signers(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).replace(";", ",").split(",") if part.strip()]


def build_credit_graph(
    frame: pd.DataFrame,
    *,
    label_column: str = "Risk_Flag",
    feature_columns: list[str] | None = None,
    syndicate_label_column: str | None = None,
) -> CreditGraph:
    """Build a PyG graph from one row per applicant.

    `employer_id`, `address_id`, hashed `phone_hash`, and
    `co_signed_applicant_id` are used when present. Missing relationship data
    gets a per-applicant placeholder, never a guessed link from occupation or
    city. No raw phone is used.
    If a syndicate label is absent, its auxiliary target is the observed default
    fraction among applicants sharing an entity or co-sign edge (self excluded).
    """
    if label_column not in frame:
        raise ValueError(f"Training data must contain label column {label_column!r}")
    frame = frame.reset_index(drop=True).copy()
    id_column = "applicant_id" if "applicant_id" in frame else "Id" if "Id" in frame else None
    if id_column is None:
        raise ValueError("Training data must contain applicant_id or Id")
    applicant_ids = [_string_key(value, f"row-{i}") for i, value in enumerate(frame[id_column])]
    if len(set(applicant_ids)) != len(applicant_ids):
        raise ValueError("Applicant identifiers must be unique per training row")

    requested = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
    usable_features = [name for name in requested if name in frame.columns]
    if not usable_features:
        raise ValueError("No numeric applicant feature columns were found in the training data")
    features = frame[usable_features].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median()).fillna(0.0)
    means = features.mean()
    scales = features.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    applicant_x = torch.tensor(((features - means) / scales).to_numpy(), dtype=torch.float32)

    indices: dict[str, dict[str, int]] = {node: {} for node in NODE_TYPES}
    indices["applicant"] = {key: idx for idx, key in enumerate(applicant_ids)}

    def node_index(node_type: str, key: str) -> int:
        if key not in indices[node_type]:
            indices[node_type][key] = len(indices[node_type])
        return indices[node_type][key]

    edges: dict[tuple[str, str, str], list[tuple[int, int]]] = {
        relation: [] for relation in EDGE_TYPES
    }
    co_sign_neighbors = [set() for _ in applicant_ids]

    for applicant_idx, row in frame.iterrows():
        employer_value = row.get("employer_id")
        employer_key = _string_key(employer_value, f"unknown-employer-{applicant_idx}")
        employer_idx = node_index("employer", employer_key)
        edges[("applicant", "employed_at", "employer")].append((applicant_idx, employer_idx))
        edges[("employer", "employs", "applicant")].append((employer_idx, applicant_idx))

        address_value = row.get("address_id")
        if address_value is None or pd.isna(address_value) or not str(address_value).strip():
            address_value = f"unknown-address-{applicant_idx}"
        address_idx = node_index("address", _string_key(address_value, f"unknown-address-{applicant_idx}"))
        edges[("applicant", "shares_address", "address")].append((applicant_idx, address_idx))
        edges[("address", "shared_by", "applicant")].append((address_idx, applicant_idx))

        phone_value = row.get("phone_hash")
        if phone_value is not None and not pd.isna(phone_value) and str(phone_value).strip():
            phone_idx = node_index("phone_hash", str(phone_value).strip())
            edges[("applicant", "shares_phone", "phone_hash")].append((applicant_idx, phone_idx))
            edges[("phone_hash", "shared_by_phone", "applicant")].append((phone_idx, applicant_idx))

    for applicant_idx, raw in enumerate(frame.get(
        "co_signed_applicant_id", pd.Series([None] * len(frame))
    )):
        for other_id in _read_co_signers(raw):
            other_idx = indices["applicant"].get(other_id)
            if other_idx is not None and other_idx != applicant_idx:
                edges[("applicant", "co_signed", "applicant")].append((applicant_idx, other_idx))
                edges[("applicant", "co_signed", "applicant")].append((other_idx, applicant_idx))
                co_sign_neighbors[applicant_idx].add(other_idx)
                co_sign_neighbors[other_idx].add(applicant_idx)

    labels = pd.to_numeric(frame[label_column], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError(f"{label_column!r} must contain binary 0/1 labels")
    label_tensor = torch.tensor(labels.to_numpy(), dtype=torch.float32)
    if syndicate_label_column and syndicate_label_column in frame:
        syndicate = pd.to_numeric(frame[syndicate_label_column], errors="coerce")
        if syndicate.isna().any() or ((syndicate < 0) | (syndicate > 1)).any():
            raise ValueError(f"{syndicate_label_column!r} must contain scores in [0, 1]")
        syndicate_tensor = torch.tensor(syndicate.to_numpy(), dtype=torch.float32)
    else:
        # Compute peer group rates in O(nodes + edges), avoiding a quadratic
        # applicant-pair expansion for large employer/address groups.
        labels_np = labels.to_numpy(dtype=np.float64)
        peer_sum = np.zeros(len(applicant_ids), dtype=np.float64)
        peer_count = np.zeros(len(applicant_ids), dtype=np.float64)
        for relation, entity_type in (
            ("employed_at", "employer"),
            ("shares_address", "address"),
            ("shares_phone", "phone_hash"),
        ):
            groups: dict[int, list[int]] = {}
            for applicant_idx, entity_idx in edges[("applicant", relation, entity_type)]:
                groups.setdefault(entity_idx, []).append(applicant_idx)
            for members in groups.values():
                if len(members) < 2:
                    continue
                group_total = float(labels_np[members].sum())
                for applicant_idx in members:
                    peer_sum[applicant_idx] += (
                        group_total - labels_np[applicant_idx]
                    ) / (len(members) - 1)
                    peer_count[applicant_idx] += 1
        for applicant_idx, neighbors in enumerate(co_sign_neighbors):
            if neighbors:
                peer_sum[applicant_idx] += float(labels_np[list(neighbors)].mean())
                peer_count[applicant_idx] += 1
        fallback_rate = float(labels_np.mean())
        values = np.where(peer_count > 0, peer_sum / np.maximum(peer_count, 1), fallback_rate)
        syndicate_tensor = torch.tensor(values, dtype=torch.float32)

    data = HeteroData()
    data["applicant"].x = applicant_x
    data["applicant"].y = label_tensor
    for node_type in NODE_TYPES[1:]:
        data[node_type].x = torch.zeros((len(indices[node_type]), len(usable_features)), dtype=torch.float32)
    for edge_type, pairs in edges.items():
        data[edge_type].edge_index = (
            torch.tensor(pairs, dtype=torch.long).t().contiguous()
            if pairs else torch.empty((2, 0), dtype=torch.long)
        )
    data["applicant"].train_mask = torch.ones(len(applicant_ids), dtype=torch.bool)
    data["applicant"].feature_mean = means.to_dict()
    data["applicant"].feature_scale = scales.to_dict()
    data["applicant"].node_keys = applicant_ids
    return CreditGraph(data, applicant_ids, usable_features, syndicate_tensor)


def train_model(
    graph: CreditGraph,
    *,
    hidden_dim: int = 32,
    epochs: int = 80,
    learning_rate: float = 0.01,
    seed: int = 42,
) -> tuple[HeteroCreditRGCN, dict[str, float]]:
    """Fit both applicant default probability and syndicate risk heads."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = HeteroCreditRGCN(graph.data["applicant"].x.size(1), hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    targets = graph.data["applicant"].y
    syndicate_targets = graph.syndicate_targets
    model.train()
    loss_value = 0.0
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(graph.data)
        default_loss = F.binary_cross_entropy_with_logits(outputs["default_logits"], targets)
        syndicate_loss = F.mse_loss(torch.sigmoid(outputs["syndicate_logits"]), syndicate_targets)
        loss = default_loss + syndicate_loss
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
    return model, {"training_loss": loss_value}


def score_applicants(model: HeteroCreditRGCN, graph: CreditGraph) -> list[dict[str, Any]]:
    model.eval()
    with torch.no_grad():
        outputs = model(graph.data)
        defaults = torch.sigmoid(outputs["default_logits"]).cpu().tolist()
        syndicate = torch.sigmoid(outputs["syndicate_logits"]).cpu().tolist()
        embeddings = outputs["embeddings"]["applicant"].cpu().tolist()
    return [
        {"applicant_id": applicant_id,
         "default_probability": float(default),
         "syndicate_risk_score": float(syndicate_score),
         "embedding": embedding}
        for applicant_id, default, syndicate_score, embedding in
        zip(graph.applicant_ids, defaults, syndicate, embeddings)
    ]


def write_scores_to_neo4j(rows: list[dict[str, Any]], *, model_version: str) -> int:
    """Persist learned embeddings and scores onto Neo4j `:applicant` nodes."""
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    if not all((uri, username, password)):
        raise RuntimeError("Set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD to export scores")
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        driver.verify_connectivity()
        database = os.getenv("NEO4J_DATABASE", "neo4j")
        driver.execute_query(
            "CREATE INDEX applicant_id_index IF NOT EXISTS FOR (a:applicant) ON (a.applicant_id)",
            database_=database,
        )
        count = 0
        for start in range(0, len(rows), 500):
            batch = rows[start:start + 500]
            records, _, _ = driver.execute_query(
                """UNWIND $rows AS row
                   MERGE (a:applicant {applicant_id: row.applicant_id})
                   SET a.rgcn_default_probability = row.default_probability,
                       a.rgcn_syndicate_risk_score = row.syndicate_risk_score,
                       a.rgcn_embedding = row.embedding,
                       a.rgcn_model_version = $model_version,
                       a.rgcn_scored_at = datetime()
                   RETURN count(a) AS written""",
                rows=batch,
                model_version=model_version,
                database_=database,
            )
            count += int(records[0]["written"]) if records else 0
        return count
    finally:
        driver.close()


@lru_cache(maxsize=1)
def _neo4j_driver():
    uri, username, password = (os.getenv("NEO4J_URI"), os.getenv("NEO4J_USERNAME"),
                               os.getenv("NEO4J_PASSWORD"))
    if not all((uri, username, password)):
        return None
    from neo4j import GraphDatabase
    return GraphDatabase.driver(
        uri,
        auth=(username, password),
        connection_timeout=3,
        connection_acquisition_timeout=3,
        max_transaction_retry_time=3,
    )


def get_applicant_graph_risk(applicant_id: str | None) -> dict[str, Any] | None:
    """Fetch the latest precomputed GNN scores; return None if unavailable."""
    if not applicant_id:
        return None
    try:
        driver = _neo4j_driver()
        if driver is None:
            return None
        records, _, _ = driver.execute_query(
            """MATCH (a:applicant {applicant_id: $applicant_id})
               RETURN a.rgcn_default_probability AS default_probability,
                      a.rgcn_syndicate_risk_score AS syndicate_risk_score,
                      a.rgcn_model_version AS model_version""",
            applicant_id=applicant_id,
            database_=os.getenv("NEO4J_DATABASE", "neo4j"),
        )
        if not records or records[0]["default_probability"] is None:
            return None
        return {
            "default_probability": float(records[0]["default_probability"]),
            "syndicate_risk_score": float(records[0]["syndicate_risk_score"]),
            "model_version": records[0]["model_version"],
        }
    except Exception:
        LOGGER.warning("Unable to retrieve R-GCN attribution for applicant", exc_info=True)
        return None
