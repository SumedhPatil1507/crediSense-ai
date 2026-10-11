"""Point-in-time credit features shared by training and online serving."""
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32

applicant = Entity(name="applicant_id", join_keys=["applicant_id"])
pan = Entity(name="pan_hash", join_keys=["pan_hash"])

credit_source = FileSource(
    name="credit_feature_events",
    path="data/credit_feature_events.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

rolling_credit_features = FeatureView(
    name="rolling_credit_features",
    entities=[applicant, pan],
    ttl=None,
    schema=[Field(name="rolling_credit_utilization", dtype=Float32)],
    source=credit_source,
    online=True,
    description="Point-in-time rolling credit utilization.",
)

payment_velocity_features = FeatureView(
    name="payment_velocity_features",
    entities=[applicant, pan],
    ttl=None,
    schema=[Field(name="payment_velocity", dtype=Float32)],
    source=credit_source,
    online=True,
    description="Point-in-time payment velocity (on-time payment fraction).",
)

debt_to_income_features = FeatureView(
    name="debt_to_income_features",
    entities=[applicant, pan],
    ttl=None,
    schema=[Field(name="debt_to_income_ratio", dtype=Float32)],
    source=credit_source,
    online=True,
    description="Point-in-time debt-to-income ratio.",
)
