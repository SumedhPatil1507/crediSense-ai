"""Redis Pub/Sub contract for real-time prediction SSE streams."""
import json
import re
from typing import Any, Mapping

_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,31}$")


def prediction_channel(symbol: str) -> str:
    """Return the canonical Redis channel for a market symbol."""
    normalized = str(symbol).strip().upper()
    if not _SYMBOL.fullmatch(normalized):
        raise ValueError("symbol must contain 1-32 letters, digits, '.', '_' or '-'")
    return f"predictions:{normalized}"


async def publish_prediction(symbol: str, event: Mapping[str, Any], *,
                             redis_url: str | None = None,
                             redis_client: Any = None) -> int:
    """Publish one JSON-safe prediction event to subscribers for ``symbol``."""
    import os
    from redis.asyncio import Redis

    owns_client = redis_client is None
    client = redis_client or Redis.from_url(
        redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    try:
        payload = json.dumps(dict(event), separators=(",", ":"), allow_nan=False)
        return int(await client.publish(prediction_channel(symbol), payload))
    finally:
        if owns_client:
            await client.aclose()
