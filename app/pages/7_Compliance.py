"""
Compliance & Security Dashboard
RBI IT Framework | DPDP Act 2023 | Encryption Status | Credit Bureau | Prometheus
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.compliance import (get_compliance_summary, RBI_CONTROLS, DPDP_CONTROLS,
                              get_bureau_score, bureau_to_risk_adjustment, BUREAU_MOCK)
from src.encryption import encryption_status
from src.model_registry import registry_storage_info
from src.metrics_server import metrics_status

st.set_page_config(layout="wide")
st.title("Compliance & Security Center")
st.caption("RBI IT Framework | DPDP Act 2023 | AES-256 Encryption | Credit Bureau | Prometheus")

tabs = st.tabs([
    "RBI IT Framework",
    "DPDP Act 2023",
    "Encryption & Security",
    "Credit Bureau Interface",
    "Infrastructure",
])

# ── TAB 1: RBI IT Framework ────────────────────────────────────────────────────
with tabs[0]:
    summary = get_compliance_summary()
    rbi = summary["rbi_it_framework"]

    st.subheader("RBI Master Direction on IT — Compliance Posture")
    c1, c2, c3 = st.columns(3)
    c1.metric("Controls Implemented", f"{rbi['implemented']}/{rbi['total']}")
    c2.metric("Partial", rbi["partial"])
    c3.metric("Compliance Score", rbi["compliance_score"])

    # Gauge chart
    score = int(rbi["compliance_score"].replace("%", ""))
    fig_gauge = px.pie(values=[score, 100-score], names=["Compliant", "Gap"],
                        color_discrete_sequence=["#28a745", "#dee2e6"],
                        hole=0.7, title="RBI IT Compliance Score")
    fig_gauge.update_layout(showlegend=False, height=250,
                              annotations=[dict(text=f"{score}%", x=0.5, y=0.5,
                                                font_size=28, showarrow=False)])
    st.plotly_chart(fig_gauge, use_container_width=True)

    df_rbi = pd.DataFrame(RBI_CONTROLS)

    def color_status(val):
        if val == "Implemented": return "background-color: #d4edda"
        if val == "Partial":     return "background-color: #fff3cd"
        return "background-color: #f8d7da"

    st.dataframe(
        df_rbi.style.map(color_status, subset=["status"]),
        use_container_width=True, hide_index=True
    )
    st.caption("Ref: RBI Master Direction on Information Technology Framework for the NBFC Sector, 2023 | https://www.rbi.org.in")

# ── TAB 2: DPDP Act 2023 ──────────────────────────────────────────────────────
with tabs[1]:
    dpdp = summary["dpdp_act_2023"]
    st.subheader("Digital Personal Data Protection Act 2023 — Compliance Posture")

    d1, d2, d3 = st.columns(3)
    d1.metric("Controls Implemented", f"{dpdp['implemented']}/{dpdp['total']}")
    d2.metric("Partial", dpdp["partial"])
    d3.metric("Compliance Score", dpdp["compliance_score"])

    df_dpdp = pd.DataFrame(DPDP_CONTROLS)
    st.dataframe(
        df_dpdp.style.map(color_status, subset=["status"]),
        use_container_width=True, hide_index=True
    )

    st.markdown("---")
    st.subheader("Data Subject Rights Implementation")
    rights = {
        "Right to Access": "GET /api/v1/predictions/{id}",
        "Right to Correction": "PATCH /api/v1/predictions/{id}",
        "Right to Erasure": "DELETE /api/v1/predictions/{id}",
        "Right to Grievance Redressal": "POST /api/v1/grievance",
        "Right to Nomination": "Documented in privacy policy",
    }
    for right, impl in rights.items():
        col_l, col_r = st.columns([1, 2])
        col_l.write(f"**{right}**")
        col_r.code(impl)

    st.caption("Ref: Digital Personal Data Protection Act 2023 — https://www.meity.gov.in/data-protection-framework")

# ── TAB 3: Encryption & Security ──────────────────────────────────────────────
with tabs[2]:
    enc = encryption_status()
    reg = registry_storage_info()

    st.subheader("AES-256 Field-Level Encryption")
    e1, e2, e3 = st.columns(3)
    e1.metric("Cryptography Library", "Available" if enc["encryption_available"] else "Not Installed")
    e2.metric("Encryption Key", "Configured" if enc["key_configured"] else "Not Set (hash fallback)")
    e3.metric("Mode", enc["mode"])

    if not enc["key_configured"]:
        st.warning("Encryption key not configured. Sensitive fields use SHA-256 hash masking (one-way). "
                   "Set `ENCRYPTION_KEY` in Streamlit secrets for full AES-256-GCM encryption.")
        with st.expander("How to configure AES-256 encryption"):
            st.markdown("""
            **Generate a strong key:**
            ```python
            import secrets
            print(secrets.token_hex(32))  # 256-bit key
            ```
            **Add to Streamlit secrets:**
            ```
            ENCRYPTION_KEY = "your-64-char-hex-key-here"
            ```
            All new predictions will be encrypted. Old records remain hashed.
            """)
    else:
        st.success("AES-256-GCM encryption active. All sensitive fields encrypted before storage.")

    st.markdown("---")
    st.subheader("PII Masking Engine")
    st.caption("Demonstrates how PII is masked before logging.")

    from src.encryption import mask_pii_strict
    demo_data = {
        "Field": ["Income", "Age", "Experience", "PAN", "Aadhaar", "Email", "Phone"],
        "Raw Value": ["12 LPA", "32 years", "5 years", "ABCDE1234F", "1234 5678 9012", "user@bank.com", "9876543210"],
        "Field Type": ["income_lpa", "age", "experience", "pan", "aadhar", "email", "phone"],
    }
    df_demo = pd.DataFrame(demo_data)
    df_demo["Masked Value"] = [
        mask_pii_strict(12, "income_lpa"),
        mask_pii_strict(32, "age"),
        mask_pii_strict(5, "experience"),
        mask_pii_strict("ABCDE1234F", "pan"),
        mask_pii_strict("123456789012", "aadhar"),
        mask_pii_strict("user@bank.com", "email"),
        mask_pii_strict("9876543210", "phone"),
    ]
    st.dataframe(df_demo, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Model Registry Storage")
    st.json(reg)
    if reg["storage"] == "local":
        st.info("Using local registry.json. Set `S3_BUCKET` + AWS credentials to enable cloud storage.")

# ── TAB 4: Credit Bureau Interface ────────────────────────────────────────────
with tabs[3]:
    st.subheader("Credit Bureau Aggregator Interface")
    st.caption(f"Mode: {'MOCK (development)' if BUREAU_MOCK else 'LIVE'} | "
               "Supports CIBIL / Experian / Equifax")

    if BUREAU_MOCK:
        st.info("Running in mock mode. Set `CREDIT_BUREAU_API_KEY` and `BUREAU_MOCK=false` for live bureau data.")

    st.markdown("---")
    st.subheader("Simulated Bureau Lookup")

    pan_input = st.text_input("Enter PAN (will be hashed before lookup)", value="DEMO12345F",
                               help="PAN is SHA-256 hashed before any API call — never sent in plaintext")

    if st.button("Fetch Bureau Score", type="primary"):
        import hashlib
        pan_hash = hashlib.sha256(pan_input.encode()).hexdigest()
        name_hash = hashlib.sha256("demo_user".encode()).hexdigest()

        with st.spinner("Querying bureau..."):
            result = get_bureau_score(pan_hash, name_hash)

        if "error" in result and result["error"]:
            st.error(f"Bureau error: {result['error']}")
        else:
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("CIBIL Score", result.get("score", "N/A"))
            r2.metric("Score Band", result.get("score_band", "N/A"))
            r3.metric("Active Loans", result.get("active_loans", "N/A"))
            r4.metric("Credit Utilization", f"{result.get('credit_utilization_pct', 'N/A')}%")

            adjustment = bureau_to_risk_adjustment(result.get("score"))
            if adjustment < 0:
                st.success(f"Bureau adjustment: {adjustment:+.0%} (score reduces predicted risk)")
            elif adjustment > 0:
                st.error(f"Bureau adjustment: {adjustment:+.0%} (score increases predicted risk)")
            else:
                st.info("No bureau adjustment applied.")

            st.json(result)

    st.markdown("---")
    st.subheader("Bureau Integration Architecture")
    st.markdown("""
    ```
    Applicant Input
         |
         v
    PAN Hash (SHA-256)  ←── PAN never sent plaintext
         |
         v
    Bureau API (CIBIL/Experian/Equifax)
         |
         v
    CIBIL Score (550-900)
         |
         v
    Risk Adjustment (-15% to +20%)
         |
         v
    Final Risk Probability = ML Score + Bureau Adjustment
    ```
    **Supported Bureaus:** CIBIL (TransUnion), Experian India, Equifax India, CRIF High Mark
    """)

# ── TAB 5: Infrastructure ─────────────────────────────────────────────────────
with tabs[4]:
    m_status = metrics_status()

    st.subheader("Prometheus Metrics")
    p1, p2 = st.columns(2)
    p1.metric("prometheus_client installed", "Yes" if m_status["prometheus_available"] else "No")
    p2.metric("Metrics Endpoint", m_status["metrics_endpoint"] or "Disabled")

    if not m_status["prometheus_enabled"]:
        st.info("Set `PROMETHEUS_ENABLED=true` in environment to expose /metrics endpoint.")
        with st.expander("Prometheus + Grafana setup"):
            st.markdown("""
            **Enable metrics:**
            ```
            PROMETHEUS_ENABLED=true
            ```

            **Prometheus scrape config:**
            ```yaml
            scrape_configs:
              - job_name: credisense
                static_configs:
                  - targets: ['your-api-host:8000']
                metrics_path: /metrics
                scrape_interval: 30s
            ```

            **Available metrics:**
            | Metric | Type | Description |
            |--------|------|-------------|
            | credisense_predictions_total | Counter | Predictions by decision/confidence |
            | credisense_prediction_latency_ms | Histogram | API latency |
            | credisense_risk_score | Histogram | Score distribution |
            | credisense_hitl_queue_pending | Gauge | HITL queue depth |
            | credisense_model_psi | Gauge | Drift PSI score |
            | credisense_feedback_total | Counter | Analyst feedback |

            Import `infra/grafana_dashboard.json` into Grafana for pre-built dashboards.
            """)
    else:
        st.success("Prometheus metrics active at /metrics")

    st.markdown("---")
    st.subheader("Deployment Architecture")
    st.markdown("""
    | Component | Current | Production Upgrade |
    |---|---|---|
    | Database | SQLite (local) | PostgreSQL / PlanetScale |
    | Model Storage | Local + S3 optional | AWS S3 / Azure Blob |
    | Task Queue | Synchronous | Redis + Celery |
    | Metrics | Optional Prometheus | Prometheus + Grafana Cloud |
    | Auth | API Key | Auth0 / Okta RBAC |
    | Deployment | Streamlit Cloud | Docker + Railway / ECS |
    | Secrets | Streamlit Secrets | AWS Secrets Manager / Vault |
    """)
