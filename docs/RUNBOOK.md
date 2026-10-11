# Local and operational runbook

## Local setup

Use Python 3.11. Create and activate a virtual environment, install
`requirements.txt` and `requirements-api.txt`, copy `.env.example` to `.env`,
and replace both placeholder secrets. Never use the sample credentials outside
an isolated local environment. Start dependencies with `docker compose up -d
postgres redis kafka`; Compose initializes `db/schema.sql` for a new PostgreSQL
volume. Start the API with
`dotenv -f .env run -- uvicorn api.main:app --reload` and the UI with
`dotenv -f .env run -- streamlit run app/app.py`. Start ingestion separately
with `dotenv -f .env run -- python -m src.streaming.kafka_consumer`.

For the Feast repo, prepare `data/credit_feature_events.parquet` per
`docs/DATA_CONTRACT.md`, then run `dotenv -f .env run -- feast apply` from the
project root. The Feast configuration reads `FEAST_REDIS_CONNECTION_STRING`;
Compose overrides it for the Redis service hostname.
Local feature definitions configure Redis at localhost; configure a protected
Redis endpoint through deployment configuration in non-local environments.
Run `python -m src.train` to retrain; this creates an ignored
`reports/generated/model_evaluation.json` with holdout metrics and artifact hashes.
The report uses a random stratified split and explicitly does not measure
temporal generalization.

## Checks and release

Run `pytest tests -q` before a change is reviewed. Run approved dependency and
container scans before release, build an immutable image, deploy to staging, run
contract/integration checks against staging services, then promote the same
image. Pin production dependency versions and record image digest, model hash,
feature definition revision, and evaluation report together.

## Signals and response

Scrape API metrics at `/metrics` when `PROMETHEUS_ENABLED=true` and consumer
metrics at port `9108`. Alert on
API error rate/latency, consumer lag, repeated processing failures, DLQ growth,
PostgreSQL/Redis/Kafka availability, and stale or missing features. On an
incident, preserve offsets and audit evidence, stop or pause consumers if
retries amplify impact, restore dependencies, replay safely from the last
committed offset, then reconcile audit rows with cache state. Keep a tested
backup/restore procedure for persistent stores.

The included compose services are for development only. Production TLS,
authentication, network policy, secret storage, retention, backup, and access
control must be supplied by the deployment environment.
