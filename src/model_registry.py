"""
Model Versioning & Registry.
Tracks model versions, metadata, and performance metrics.
"""
import json
import hashlib
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[1]
REGISTRY_PATH = BASE_DIR / "models" / "registry.json"


def _load_registry() -> list:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return []


def _save_registry(registry: list):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def register_model(model_path: str, metrics: dict, description: str = "") -> str:
    """Register a model version. Returns version ID."""
    registry = _load_registry()
    with open(model_path, "rb") as f:
        model_hash = hashlib.md5(f.read()).hexdigest()[:12]

    version_id = f"v{len(registry) + 1}.0"
    entry = {
        "version_id": version_id,
        "timestamp": datetime.utcnow().isoformat(),
        "model_path": str(model_path),
        "model_hash": model_hash,
        "metrics": metrics,
        "description": description,
        "status": "active",
    }
    # Retire previous active versions
    for e in registry:
        if e["status"] == "active":
            e["status"] = "retired"
    registry.append(entry)
    _save_registry(registry)
    return version_id


def get_active_version() -> dict | None:
    registry = _load_registry()
    for entry in reversed(registry):
        if entry["status"] == "active":
            return entry
    return None


def get_all_versions() -> list:
    return _load_registry()


def get_current_model_hash() -> str:
    model_path = BASE_DIR / "models" / "model.pkl"
    if not model_path.exists():
        return "unknown"
    with open(model_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:12]
