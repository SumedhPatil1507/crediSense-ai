# Data and feature contracts

## Application event (`credit.applications.v1`)

UTF-8 JSON object with `application_id` (stable unique string), `applicant_id`
(stable non-PII internal identifier), `pan` (required at ingress; never written
raw to the audit log, cache, application log, or dead-letter event), and an
ISO-8601 timezone-aware `event_timestamp`. Numeric fields:

| Field | Meaning | Accepted range |
|---|---|---:|
| `rolling_credit_utilization` | Current rolling balance / available credit | 0–10 |
| `payment_velocity` | On-time payment fraction in the agreed lookback window | 0–1 |
| `debt_to_income_ratio` | Monthly debt payments / monthly gross income | 0–100 |

Send UTC timestamps with an explicit `Z` or offset. `event_timestamp` means the
time the underlying financial event was effective, not the ingestion time.
Feature event producers must also populate Feast `created_timestamp` with when
the system first knew the value; this supports correction-aware historical
queries. The feature repository's event file is generated/loaded externally and
is intentionally ignored by git because it may contain sensitive data.

`application_id` is the PostgreSQL audit idempotency key. Feast manages online
feature keys in its configured Redis store; the consumer also keeps a small
timestamp guard per pseudonymous applicant ID to prevent late events from
regressing the latest event-time snapshot. Guard state is stored under
`feast:feature-latest:{applicant_id}`; the short-lived lock is stored under
`feast:feature-lock:{applicant_id}`. Define and implement a retention/cleanup
policy for guard and audit data before production use. The Kafka consumer writes the audit row first,
then all three Feast online views, then commits the offset;
redelivery after a partial failure is expected and must remain safe. Invalid
events go to `credit.applications.v1.dlq` with PAN/SSN removed and are committed
only after the broker confirms the DLQ write.

This contract is a project interface, not a regulatory definition. Confirm
lookback windows, ratio units, retention, data residency, and deletion policy
with the data owner before production use.
