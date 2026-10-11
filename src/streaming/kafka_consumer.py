"""Kafka ingestion with an explicit audit/cache-before-commit boundary."""
import asyncio
import hashlib
import hmac
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from aiokafka import AIOKafkaConsumer
from aiokafka import AIOKafkaProducer
from aiokafka.structs import OffsetAndMetadata, TopicPartition
import asyncpg
import pandas as pd
from feast import FeatureStore
from redis.asyncio import Redis
from src.streaming.contracts import InvalidApplication, redact_sensitive, validate_application
from src.streaming.metrics import (MESSAGES_PROCESSED, MESSAGES_DLQ,
                                   PROCESSING_SECONDS, PROCESSING_FAILURES,
                                   CONSUMER_LAG)

LOGGER = logging.getLogger(__name__)
TOPIC = "credit.applications.v1"
DEAD_LETTER_TOPIC = "credit.applications.v1.dlq"
_ACQUIRE_FEATURE_LOCK = """
if redis.call('EXISTS', KEYS[2]) == 1 then return -1 end
local latest = redis.call('HGET', KEYS[1], 'latest')
if latest and ARGV[1] < latest then return 0 end
redis.call('SET', KEYS[2], ARGV[2], 'PX', ARGV[3])
return 1
"""
_FINALIZE_FEATURE_LOCK = """
if redis.call('GET', KEYS[2]) ~= ARGV[2] then return 0 end
redis.call('HSET', KEYS[1], 'latest', ARGV[1])
redis.call('DEL', KEYS[2])
return 1
"""
_RELEASE_FEATURE_LOCK = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  redis.call('DEL', KEYS[1])
  return 1
end
return 0
"""
FEATURES_BY_VIEW = {
    "rolling_credit_features": "rolling_credit_utilization",
    "payment_velocity_features": "payment_velocity",
    "debt_to_income_features": "debt_to_income_ratio",
}


def _pan_hash(value: str) -> str:
    """Hash PAN with a deployment secret so raw PAN never enters the cache/log."""
    secret = os.environ.get("PAN_HASH_SECRET")
    if not secret or len(secret.encode()) < 32:
        raise RuntimeError("PAN_HASH_SECRET must contain at least 32 bytes of random secret material")
    return hmac.new(secret.encode(), value.strip().upper().encode(), hashlib.sha256).hexdigest()


class LoanApplicationConsumer:
    def __init__(self, *, consumer: Any = None, pool: Any = None,
                 redis: Any = None, producer: Any = None,
                 feature_store: Any = None, topic: str = TOPIC):
        self.topic = topic
        self.consumer = consumer or AIOKafkaConsumer(
            topic,
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            group_id=os.getenv("KAFKA_GROUP_ID", "credisense-credit-applications"),
            enable_auto_commit=False,
            value_deserializer=lambda raw: json.loads(raw.decode("utf-8")),
        )
        self.pool = pool
        self.producer = producer or AIOKafkaProducer(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"))
        self.redis = redis or Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                                             decode_responses=True)
        self.feature_store = feature_store or FeatureStore(
            repo_path=os.getenv("FEAST_REPO_PATH", str(Path(__file__).resolve().parents[2])))
        self._owns_pool = pool is None
        self._owns_redis = redis is None

    async def start(self) -> None:
        if self.pool is None:
            dsn = os.getenv("DATABASE_URL")
            if not dsn:
                raise RuntimeError("DATABASE_URL must be set for application audit logging")
            self.pool = await asyncpg.create_pool(dsn)
        await self.consumer.start()
        await self.producer.start()

    async def stop(self) -> None:
        await self.consumer.stop()
        await self.producer.stop()
        if self._owns_pool and self.pool is not None:
            await self.pool.close()
        if self._owns_redis:
            await self.redis.aclose()

    async def process_application(self, payload: Mapping[str, Any]) -> str:
        """Persist audit record then publish features; errors prevent offset commit."""
        started = asyncio.get_running_loop().time()
        payload = validate_application(payload)
        application_id = str(payload["application_id"])
        applicant_id = str(payload["applicant_id"])
        pan_digest = _pan_hash(str(payload["pan"]))
        event_time = payload.get("event_timestamp") or datetime.now(timezone.utc).isoformat()
        feature_values = {
            "rolling_credit_utilization": float(payload["rolling_credit_utilization"]),
            "payment_velocity": float(payload["payment_velocity"]),
            "debt_to_income_ratio": float(payload["debt_to_income_ratio"]),
        }
        sanitized = redact_sensitive(payload)
        sanitized["pan_hash"] = pan_digest
        async with self.pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO loan_application_audit
                   (application_id, applicant_id, pan_hash, event_timestamp, payload)
                   VALUES ($1, $2, $3, $4, $5::jsonb)
                   ON CONFLICT (application_id) DO NOTHING""",
                application_id, applicant_id, pan_digest, event_time,
                json.dumps(sanitized),
            )
        # Serialize updates per applicant and reject late events from regressing
        # Feast's online value. Audit history still retains every accepted event.
        guard_key = f"feast:feature-latest:{applicant_id}"
        lock_key = f"feast:feature-lock:{applicant_id}"
        lock_token = f"{application_id}:{uuid.uuid4().hex}"
        acquired = await self.redis.eval(
            _ACQUIRE_FEATURE_LOCK, 2, guard_key, lock_key, event_time, lock_token, 120_000)
        if acquired == 0:
            PROCESSING_SECONDS.observe(asyncio.get_running_loop().time() - started)
            MESSAGES_PROCESSED.inc()
            return applicant_id
        if acquired != 1:
            raise RuntimeError("Feature update is busy for this applicant; retrying")
        event_row = {"applicant_id": applicant_id, "pan_hash": pan_digest,
                     "event_timestamp": event_time,
                     "created_timestamp": datetime.now(timezone.utc).isoformat(),
                     **feature_values}
        try:
            writes = [asyncio.to_thread(
                self.feature_store.write_to_online_store,
                view_name,
                pd.DataFrame([{**{k: event_row[k] for k in
                                ("applicant_id", "pan_hash", "event_timestamp", "created_timestamp")},
                               field_name: event_row[field_name]}]),
            ) for view_name, field_name in FEATURES_BY_VIEW.items()]
            write_results = await asyncio.gather(*writes, return_exceptions=True)
            failures = [result for result in write_results if isinstance(result, Exception)]
            if failures:
                raise failures[0]
            finalized = await self.redis.eval(
                _FINALIZE_FEATURE_LOCK, 2, guard_key, lock_key, event_time, lock_token)
            if finalized != 1:
                raise RuntimeError("Could not finalize Feast online feature write")
        except Exception:
            await self.redis.eval(_RELEASE_FEATURE_LOCK, 1, lock_key, lock_token)
            raise
        PROCESSING_SECONDS.observe(asyncio.get_running_loop().time() - started)
        MESSAGES_PROCESSED.inc()
        return applicant_id

    async def run(self) -> None:
        await self.start()
        try:
            async for message in self.consumer:
                try:
                    for attempt in range(3):
                        try:
                            await self.process_application(message.value)
                            break
                        except InvalidApplication as exc:
                            # Route poison events to DLQ before acknowledging their source offset.
                            raw_event = message.value
                            dead_letter = {"source_topic": message.topic,
                                           "source_partition": message.partition,
                                           "source_offset": message.offset,
                                           "reason": str(exc),
                                           "event": redact_sensitive(raw_event)
                                           if isinstance(raw_event, Mapping) else None,
                                           "event_type": type(raw_event).__name__}
                            await self.producer.send_and_wait(
                                DEAD_LETTER_TOPIC, json.dumps(dead_letter).encode("utf-8"))
                            MESSAGES_DLQ.inc()
                            break
                        except Exception:
                            PROCESSING_FAILURES.inc()
                            if attempt == 2:
                                LOGGER.exception("Application processing failed; offset remains uncommitted")
                                raise
                            await asyncio.sleep(0.25 * (2 ** attempt))
                    # Commit only after audit + Redis, or after durable DLQ publication.
                    partition = TopicPartition(message.topic, message.partition)
                    await self.consumer.commit({partition: OffsetAndMetadata(message.offset + 1, "")})
                    try:
                        end_offset = (await self.consumer.end_offsets([partition]))[partition]
                        CONSUMER_LAG.labels(message.topic, str(message.partition)).set(
                            max(0, end_offset - message.offset - 1))
                    except Exception:
                        LOGGER.debug("Unable to refresh consumer lag metric", exc_info=True)
                except Exception:
                    LOGGER.exception("Kafka message was not acknowledged")
                    raise
        finally:
            await self.stop()


async def main() -> None:
    from prometheus_client import start_http_server
    start_http_server(int(os.getenv("STREAM_METRICS_PORT", "9108")),
                      addr=os.getenv("STREAM_METRICS_HOST", "0.0.0.0"))
    await LoanApplicationConsumer().run()


if __name__ == "__main__":
    asyncio.run(main())
