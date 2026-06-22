"""
Model Registry with S3-compatible cloud storage and local JSON fallback.
Uses boto3 for S3/MinIO/Azure Blob if S3_BUCKET env var is set.
Falls back to local registry.json transparently.

Cloud setup:
  AWS S3:   Set S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
  MinIO:    Set S3_BUCKET, S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
  Azure:    Use azurite-compatible endpoint via S3_ENDPOINT_URL
"""
import json
import hashlib
import os
from pathlib import Path
from datetime import datetime

BASE_DIR      = Path(__file__).resolve().parents[1]
REGISTRY_PATH = BASE_DIR / "models" / "registry.json"

# S3 config
S3_BUCKET  = os.getenv("S3_BUCKET", "")
S3_PREFIX  = os.getenv("S3_PREFIX", "credisense/models/")
S3_ENDPOINT = os.getenv("S3_ENDPOINT_URL", "")

REGISTRY_S3_KEY = f"{S3_PREFIX}registry.json"


def _s3_client():
    """Return boto3 S3 client if configured, else None."""
    if not S3_BUCKET:
        return None
    try:
        import boto3
        kwargs = {}
        if S3_ENDPOINT:
            kwargs["endpoint_url"] = S3_ENDPOINT
        return boto3.client("s3", **kwargs)
    except Exception:
        return None


def _load_registry() -> list:
    """Load registry from S3 if available, else local file."""
    s3 = _s3_client()
    if s3:
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=REGISTRY_S3_KEY)
            return json.loads(obj["Body"].read())
        except Exception:
            pass  # Fall through to local
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return []


def _save_registry(registry: list):
    """Save registry to S3 if available, always save locally too."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    s3 = _s3_client()
    if s3:
        try:
            s3.put_object(
                Bucket=S3_BUCKET,
                Key=REGISTRY_S3_KEY,
                Body=json.dumps(registry, indent=2).encode(),
                ContentType="application/json",
            )
        except Exception:
            pass  # Local backup already written


def _upload_model_to_s3(model_path: str, version_id: str) -> str | None:
    """Upload model artifact to S3. Returns S3 URI or None."""
    s3 = _s3_client()
    if not s3:
        return None
    s3_key = f"{S3_PREFIX}{version_id}/model.pkl"
    try:
        s3.upload_file(model_path, S3_BUCKET, s3_key)
        return f"s3://{S3_BUCKET}/{s3_key}"
    except Exception:
        return None


def register_model(model_path: str, metrics: dict, description: str = "") -> str:
    """Register a model version. Uploads to S3 if configured. Returns version ID."""
    registry = _load_registry()

    with open(model_path, "rb") as f:
        model_hash = hashlib.sha256(f.read()).hexdigest()[:16]  # SHA-256 (stronger than MD5)

    version_id = f"v{len(registry) + 1}.0"

    # Upload to cloud if configured
    s3_uri = _upload_model_to_s3(model_path, version_id)

    entry = {
        "version_id": version_id,
        "timestamp": datetime.utcnow().isoformat(),
        "model_path": s3_uri if s3_uri else str(model_path),
        "model_hash": model_hash,
        "hash_algo": "sha256",
        "metrics": metrics,
        "description": description,
        "status": "active",
        "storage": "s3" if s3_uri else "local",
        "s3_uri": s3_uri,
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
        return hashlib.sha256(f.read()).hexdigest()[:16]


def registry_storage_info() -> dict:
    s3 = _s3_client()
    return {
        "storage": "s3" if s3 else "local",
        "s3_bucket": S3_BUCKET if S3_BUCKET else None,
        "s3_prefix": S3_PREFIX,
        "local_path": str(REGISTRY_PATH),
    }
