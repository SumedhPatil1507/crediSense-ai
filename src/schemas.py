"""
Strict Pydantic v2 schemas for all input/output validation.
Used by both the FastAPI backend and Streamlit pages.
"""
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional
from datetime import datetime


class ApplicantInput(BaseModel):
    """Validated applicant input — real-world values, not normalized."""
    income_lpa: float = Field(..., gt=0, le=500,
                               description="Annual income in LPA (lakhs per annum)")
    age_years: int    = Field(..., ge=18, le=70, description="Age in years")
    experience_years: int = Field(..., ge=0, le=45,
                                   description="Total work experience in years")
    profession: str   = Field(default="Engineer", max_length=100)
    city: str         = Field(default="Mumbai", max_length=100)
    state: str        = Field(default="Maharashtra", max_length=100)
    house_ownership: str  = Field(default="owned",
                                   pattern="^(owned|rented|norent_noown)$")
    marital_status: str   = Field(default="single", pattern="^(single|married)$")
    car_ownership: str    = Field(default="no", pattern="^(yes|no)$")
    current_job_years: int   = Field(default=2, ge=0, le=40)
    current_house_years: int = Field(default=3, ge=0, le=40)

    @field_validator("experience_years")
    @classmethod
    def experience_lt_working_age(cls, v: int, info) -> int:
        age = info.data.get("age_years", 70)
        if v >= age - 16:
            raise ValueError(
                f"Experience ({v} yrs) cannot exceed working age "
                f"(age {age} - 16 = {age - 16} max)"
            )
        return v

    @model_validator(mode="after")
    def job_years_lt_experience(self) -> "ApplicantInput":
        if self.current_job_years > self.experience_years:
            raise ValueError(
                f"Current job years ({self.current_job_years}) "
                f"cannot exceed total experience ({self.experience_years})"
            )
        return self

    def normalize(self) -> dict:
        """Return normalized (0-1) values for model input."""
        return {
            "income_norm": min(self.income_lpa / 50.0, 1.0),
            "age_norm":    (self.age_years - 18) / 52.0,
            "exp_norm":    min(self.experience_years / 40.0, 1.0),
        }


class PredictionOutput(BaseModel):
    """Structured prediction response."""
    model_config = {"protected_namespaces": ()}

    prediction_id: str
    risk_probability: float = Field(..., ge=0.0, le=1.0)
    safe_probability: float = Field(..., ge=0.0, le=1.0)
    decision: str = Field(..., pattern="^(Approve|Manual Review|Reject)$")
    confidence: str
    confidence_interval_95: dict
    adverse_action: Optional[dict] = None
    model_version: str
    model_hash: str
    queued_for_review: bool
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class FeedbackInput(BaseModel):
    """Analyst feedback on a prediction."""
    prediction_id: str = Field(..., min_length=1)
    feedback: str = Field(..., pattern="^(correct|incorrect|unsure)$")
    corrected_label: Optional[str] = Field(default="", max_length=100)
    notes: Optional[str] = Field(default="", max_length=500)


class BatchInput(BaseModel):
    """Batch prediction request."""
    applicants: list[ApplicantInput] = Field(..., min_length=1, max_length=500)


class DriftReport(BaseModel):
    """PSI drift monitoring report."""
    psi_score: float
    status: str = Field(..., pattern="^(stable|warning|critical)$")
    message: str
    training_samples: int
    live_samples: int
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
