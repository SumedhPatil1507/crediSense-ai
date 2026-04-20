"""
LLM module — rule-based fallback only.
LLM integration removed due to API key issues.
"""


def explain_prediction(prob: float, income: float, age: float, experience: float,
                       shap_top=None) -> str:
    """Rule-based plain-English explanation of a credit risk prediction."""
    decision = "approved" if prob < 0.3 else "flagged for manual review" if prob < 0.6 else "rejected"
    risk_level = "low" if prob < 0.3 else "moderate" if prob < 0.6 else "high"

    drivers = []
    if income < 0.3:
        drivers.append("below-average income")
    if experience < 0.2:
        drivers.append("limited work experience")
    if age < 0.2:
        drivers.append("young applicant profile")
    if income > 0.7 and experience > 0.5:
        drivers.append("strong income and experience profile")

    driver_text = f" Key factors: {', '.join(drivers)}." if drivers else ""
    return (f"This applicant has a {risk_level} default risk with a probability of {prob:.1%}.{driver_text} "
            f"The application is {decision} based on the model assessment.")


def chat_with_analyst(messages: list) -> str:
    """Rule-based credit risk Q&A fallback."""
    if not messages:
        return "Ask me anything about credit risk or loan decisions."

    last = messages[-1]["content"].lower()

    if any(w in last for w in ["what is", "explain", "define"]):
        if "credit risk" in last:
            return ("Credit risk is the probability that a borrower will default on a loan. "
                    "It's assessed using factors like income, employment stability, and repayment history.")
        if "gini" in last:
            return "Gini coefficient = 2×AUC - 1. A score of 0.7+ is considered good for credit models."
        if "ks" in last or "kolmogorov" in last:
            return ("KS (Kolmogorov-Smirnov) statistic measures the maximum separation between "
                    "the cumulative distributions of defaulters and non-defaulters. Higher is better.")

    if any(w in last for w in ["threshold", "cutoff"]):
        return ("Lowering the threshold catches more defaults (higher recall) but increases false rejections. "
                "Raising it approves more applicants but misses some risky ones. "
                "Typical credit models use 0.3–0.5 depending on risk appetite.")

    if any(w in last for w in ["shap", "feature", "important"]):
        return ("SHAP values show each feature's contribution to a prediction. "
                "Positive SHAP = increases risk, negative = decreases risk. "
                "Income and experience are typically the strongest drivers in this model.")

    if any(w in last for w in ["approve", "reject", "decision"]):
        return ("This model uses a 3-tier decision: Approve (risk < 30%), "
                "Manual Review (30–60%), Reject (> 60%). "
                "Thresholds can be adjusted based on the lender's risk appetite.")

    return ("I can answer questions about credit risk metrics, model decisions, SHAP explanations, "
            "and threshold tuning. What would you like to know?")
