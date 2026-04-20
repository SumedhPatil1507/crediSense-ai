"""
LLM integration via Groq API (free tier, llama3-8b-8192).
Set GROQ_API_KEY in Streamlit secrets or environment variable.
"""
import os
import streamlit as st


def get_groq_client():
    try:
        from groq import Groq
        api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
        if not api_key:
            return None
        return Groq(api_key=api_key)
    except Exception:
        return None


def explain_prediction(prob: float, income: float, age: float, experience: float,
                        shap_top: list[dict] | None = None) -> str:
    """
    Generate a plain-English explanation of a credit risk prediction.
    shap_top: list of {"feature": str, "value": float} sorted by abs importance.
    """
    client = get_groq_client()
    if client is None:
        return _fallback_explanation(prob, income, age, experience)

    decision = "Approve" if prob < 0.3 else "Manual Review" if prob < 0.6 else "Reject"
    shap_text = ""
    if shap_top:
        top3 = shap_top[:3]
        shap_text = "Top contributing factors: " + ", ".join(
            f"{d['feature']} (impact: {d['value']:+.3f})" for d in top3
        )

    prompt = f"""You are a credit risk analyst at a bank. Explain this loan application assessment in 3-4 clear sentences for a loan officer.

Applicant data (normalized 0-1):
- Income: {income:.2f}
- Age: {age:.2f}  
- Experience: {experience:.2f}
- Risk Probability: {prob:.1%}
- Decision: {decision}
{shap_text}

Write a professional, concise explanation. Mention the key risk drivers. End with the recommendation."""

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return _fallback_explanation(prob, income, age, experience)


def parse_natural_language_input(text: str) -> dict | None:
    """
    Parse free-form text like '35 year old engineer earning 8 LPA with 5 years experience'
    into normalized {Income, Age, Experience} dict.
    Returns None if parsing fails.
    """
    client = get_groq_client()
    if client is None:
        return None

    prompt = f"""Extract loan applicant details from this text and return ONLY a JSON object.

Text: "{text}"

Rules:
- Income: normalize to 0-1 range (assume max income = 50 LPA, so 8 LPA = 0.16)
- Age: normalize to 0-1 range (assume range 18-70, so 35 years = (35-18)/(70-18) = 0.33)
- Experience: normalize to 0-1 range (assume max = 40 years, so 5 years = 0.125)
- If a value is missing, use 0.5 as default

Return ONLY valid JSON like: {{"Income": 0.16, "Age": 0.33, "Experience": 0.125}}
No explanation, no markdown, just the JSON."""

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1,
        )
        import json
        raw = response.choices[0].message.content.strip()
        # Extract JSON from response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except Exception:
        return None


def chat_with_analyst(messages: list[dict]) -> str:
    """
    General credit risk Q&A. messages = [{"role": "user"/"assistant", "content": str}]
    """
    client = get_groq_client()
    if client is None:
        return "LLM not available. Please set GROQ_API_KEY in Streamlit secrets."

    system = """You are CrediSense AI, an expert credit risk analyst assistant. 
You help loan officers understand credit risk, interpret model predictions, and make lending decisions.
Keep answers concise (2-4 sentences). Focus on practical credit risk insights."""

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "system", "content": system}] + messages,
            max_tokens=300,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"


def _fallback_explanation(prob: float, income: float, age: float, experience: float) -> str:
    """Rule-based fallback when LLM is unavailable."""
    decision = "approved" if prob < 0.3 else "flagged for manual review" if prob < 0.6 else "rejected"
    risk_level = "low" if prob < 0.3 else "moderate" if prob < 0.6 else "high"
    drivers = []
    if income < 0.3:
        drivers.append("below-average income")
    if experience < 0.2:
        drivers.append("limited work experience")
    if age < 0.2:
        drivers.append("young applicant profile")
    driver_text = f" Key risk drivers: {', '.join(drivers)}." if drivers else ""
    return (f"This applicant has a {risk_level} default risk with a probability of {prob:.1%}.{driver_text} "
            f"The application is {decision} based on the model assessment.")
