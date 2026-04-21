"""
Adverse Action Notice generator.
Produces regulation-style rejection reasons based on model inputs and SHAP values.
Inspired by ECOA/FCRA adverse action notice requirements.
"""


REASON_TEMPLATES = {
    "income":      "Insufficient income relative to loan amount",
    "experience":  "Insufficient length of employment history",
    "age":         "Insufficient credit history length (age proxy)",
    "stability":   "Insufficient stability in current residence/employment",
    "income_exp":  "Debt-to-income ratio too high given experience level",
}


def generate_adverse_action(prob: float, income: float, age: float,
                             experience: float, shap_top: list[dict] | None = None) -> dict:
    """
    Returns a structured adverse action notice dict.
    shap_top: [{"feature": str, "shap_value": float}, ...] sorted by abs importance.
    """
    if prob < 0.6:
        return {"required": False, "decision": "Approve / Review", "reasons": []}

    reasons = []

    # SHAP-based reasons (most accurate)
    if shap_top:
        for item in shap_top[:3]:
            feat = item["feature"].replace("num__", "").replace("cat__", "")
            val = item["shap_value"]
            if val > 0.01:  # positive SHAP = increases risk
                if "Income" in feat:
                    reasons.append(REASON_TEMPLATES["income"])
                elif "Experience" in feat or "JOB" in feat.upper():
                    reasons.append(REASON_TEMPLATES["experience"])
                elif "Age" in feat:
                    reasons.append(REASON_TEMPLATES["age"])
                elif "stability" in feat.lower() or "HOUSE" in feat.upper():
                    reasons.append(REASON_TEMPLATES["stability"])
    
    # Rule-based fallback
    if not reasons:
        if income < 0.25:
            reasons.append(REASON_TEMPLATES["income"])
        if experience < 0.15:
            reasons.append(REASON_TEMPLATES["experience"])
        if age < 0.2:
            reasons.append(REASON_TEMPLATES["age"])
        if not reasons:
            reasons.append("Overall risk profile exceeds acceptable threshold")

    # Deduplicate
    reasons = list(dict.fromkeys(reasons))[:3]

    return {
        "required": True,
        "decision": "Application Declined",
        "risk_score": round(prob, 4),
        "reasons": reasons,
        "notice": (
            "ADVERSE ACTION NOTICE\n"
            "Your loan application has been declined based on information "
            "obtained from our credit risk model. The principal reasons are:\n"
            + "\n".join(f"  {i+1}. {r}" for i, r in enumerate(reasons))
            + "\n\nYou have the right to request the specific reasons for this decision."
        ),
        "citation": "Ref: Equal Credit Opportunity Act (ECOA) · Fair Credit Reporting Act (FCRA)"
    }
