"""Prometheus metrics for credit application ingestion."""
from prometheus_client import Counter, Gauge, Histogram

MESSAGES_PROCESSED = Counter("credit_stream_messages_processed_total", "Successfully processed applications")
MESSAGES_DLQ = Counter("credit_stream_messages_dlq_total", "Invalid applications routed to the DLQ")
PROCESSING_SECONDS = Histogram("credit_stream_processing_seconds", "Application processing latency")
PROCESSING_FAILURES = Counter("credit_stream_processing_failures_total", "Processing attempts that failed")
CONSUMER_LAG = Gauge("credit_stream_consumer_lag_messages", "Unprocessed source records", ["topic", "partition"])
