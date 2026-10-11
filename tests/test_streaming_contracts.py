import pytest

from src.streaming.contracts import InvalidApplication, redact_sensitive, validate_application


def _event(**updates):
    event = {
        "application_id": "app-1", "applicant_id": "person-1", "pan": "ABCDE1234F",
        "event_timestamp": "2025-04-01T12:00:00Z",
        "rolling_credit_utilization": 0.45, "payment_velocity": 0.92,
        "debt_to_income_ratio": 0.31,
    }
    event.update(updates)
    return event


def test_application_contract_normalizes_timestamp_and_numbers():
    event = validate_application(_event())
    assert event["event_timestamp"].endswith("+00:00")
    assert event["payment_velocity"] == pytest.approx(0.92)


@pytest.mark.parametrize("timestamp", ["2025-04-01T12:00:00", "yesterday"])
def test_rejects_missing_timezone_or_invalid_timestamp(timestamp):
    with pytest.raises(InvalidApplication):
        validate_application(_event(event_timestamp=timestamp))


@pytest.mark.parametrize("field,value", [
    ("payment_velocity", 1.1), ("rolling_credit_utilization", -0.1),
    ("debt_to_income_ratio", float("nan")),
])
def test_rejects_out_of_range_or_non_finite_features(field, value):
    with pytest.raises(InvalidApplication):
        validate_application(_event(**{field: value}))


def test_rejects_missing_required_fields():
    event = _event()
    del event["applicant_id"]
    with pytest.raises(InvalidApplication, match="applicant_id"):
        validate_application(event)


def test_redacts_sensitive_fields_recursively():
    event = {"pan": "ABCDE1234F", "metadata": {"account_number": "1234"},
             "payment_velocity": 0.9}
    assert redact_sensitive(event) == {
        "pan": "[REDACTED]", "metadata": {"account_number": "[REDACTED]"},
        "payment_velocity": 0.9,
    }
