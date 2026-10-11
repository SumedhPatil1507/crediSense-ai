"""Validation for versioned credit application events."""
from datetime import datetime, timezone
from math import isfinite
from typing import Any, Mapping


class InvalidApplication(ValueError):
    """An event is malformed or outside the documented domain contract."""


def validate_application(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise InvalidApplication("Event must be a JSON object")
    required = ("application_id", "applicant_id", "pan", "event_timestamp",
                "rolling_credit_utilization", "payment_velocity", "debt_to_income_ratio")
    missing = [key for key in required if payload.get(key) in (None, "")]
    if missing:
        raise InvalidApplication(f"Missing required fields: {', '.join(missing)}")
    try:
        timestamp = datetime.fromisoformat(str(payload["event_timestamp"]).replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            raise ValueError("timezone required")
    except (TypeError, ValueError) as exc:
        raise InvalidApplication("event_timestamp must be ISO-8601 with timezone") from exc
    values = {}
    bounds = {"rolling_credit_utilization": (0.0, 10.0),
              "payment_velocity": (0.0, 1.0), "debt_to_income_ratio": (0.0, 100.0)}
    for name, (low, high) in bounds.items():
        try:
            value = float(payload[name])
        except (TypeError, ValueError) as exc:
            raise InvalidApplication(f"{name} must be numeric") from exc
        if not isfinite(value) or not low <= value <= high:
            raise InvalidApplication(f"{name} must be between {low} and {high}")
        values[name] = value
    if not str(payload["pan"]).strip():
        raise InvalidApplication("pan must not be blank")
    result = dict(payload)
    result.update(values)
    result["event_timestamp"] = timestamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
    return result


def redact_sensitive(value: Any) -> Any:
    """Recursively remove common direct identifiers before persisting event data."""
    sensitive = {"pan", "ssn", "aadhaar", "national_id", "account_number",
                 "email", "email_address", "phone", "phone_number", "mobile_number",
                 "date_of_birth", "dob", "address"}
    if isinstance(value, Mapping):
        return {str(key): "[REDACTED]" if str(key).lower() in sensitive
                else redact_sensitive(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    return value
