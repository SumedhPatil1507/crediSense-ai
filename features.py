"""Feast CLI entry point; definitions are maintained in feature_repo."""
from feature_repo.features import (  # noqa: F401
    applicant,
    debt_to_income_features,
    pan,
    payment_velocity_features,
    rolling_credit_features,
)
