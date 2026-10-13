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

# Market events have both event time and arrival/creation time. Feast uses the
# former for point-in-time joins and the latter to exclude late-arriving data
# that was not yet available at the historical decision cutoff.
symbol = Entity(name="symbol", join_keys=["symbol"])
market_source = FileSource(
    name="market_feature_events",
    path="data/market_feature_events.parquet",
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

market_microstructure_features = FeatureView(
    name="market_microstructure_features",
    entities=[symbol],
    ttl=None,
    schema=[
        Field(name="order_flow_imbalance", dtype=Float32),
        Field(name="vwap_slippage_bps", dtype=Float32),
        Field(name="bid_ask_spread_bps", dtype=Float32),
    ],
    source=market_source,
    online=True,
    description="Event-time order book imbalance, execution slippage and spread.",
)

market_technical_features = FeatureView(
    name="market_technical_features",
    entities=[symbol],
    ttl=None,
    schema=[
        Field(name="return_1", dtype=Float32),
        Field(name="return_5", dtype=Float32),
        Field(name="realized_volatility_20", dtype=Float32),
        Field(name="rsi_14", dtype=Float32),
    ],
    source=market_source,
    online=True,
    description="Trailing close-based indicators computed only from prior bars.",
)
