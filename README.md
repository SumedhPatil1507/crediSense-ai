# CrediSense AI: Advanced Enterprise Credit Risk Platform

A production-grade, end-to-end **Credit Risk Scoring System** built to advanced enterprise standards. Designed to predict loan default probability with uncompromised explainability, regulatory compliance, real-time MLOps infrastructure, and algorithmic fairness auditing.

**Live Demo:** [Streamlit Cloud Demo Link](https://credisense-ai.streamlit.app/)

![CI](https://github.com/SumedhPatil1507/crediSense-ai/actions/workflows/ci.yml/badge.svg)

---

## 🌟 Next-Gen Enterprise Features

| Capability | Implementation |
|---|---|
| **Macro Stress Testing** | Portfolio-level simulation of macroeconomic shocks (e.g., income drops, unemployment spikes) |
| **Algorithmic Fairness** | Fairlearn integration (Demographic Parity & Disparate Impact) across protected attributes |
| **Agentic Human-in-the-Loop** | AI Agent pre-analyzes borderline cases and proposes a recommended decision + rationale |
| **Self-Healing MLOps** | Automated model retraining triggers when critical data drift (PSI) is detected |
| **Database Persistence** | Strict integration with Supabase (PostgreSQL) for enterprise auditability |
| **ML Model** | LightGBM with hyperparameter tuning, class imbalance handling |
| **Explainability** | SHAP (global + waterfall + dependence) + LIME (local) |
| **Shadow Mode** | Run challenger model silently alongside champion |
| **Regulatory Compliance** | ECOA/FCRA adverse action notices, audit trail, PII masking |

---

## 🏗️ System Architecture

```
User/Client
    |
    v
FastAPI Backend (api/main.py)          Streamlit Dashboard (app/)
    |                                       |
    +-- /predict (single)                   +-- 1_Stress_Testing.py (Macro shock simulation)
    +-- /predict/batch (up to 500)          +-- 2_Model.py (Predict + evaluate + what-if)
    +-- /explain (SHAP top features)        +-- 3_Explainability.py (SHAP + LIME)
    +-- /feedback (analyst corrections)     +-- 4_Chatbot.py (Risk assistant + Q&A)
    +-- /metrics (aggregate stats)          +-- 5_Logs.py (Logs + audit + cost-benefit)
    +-- /health (system status)             +-- 6_Operations.py (Agentic HITL + Self-Healing)
    |                                       +-- 7_Fairness.py (Bias & Fairness Audits)
    v
Supabase (PostgreSQL)
    |
    +-- predictions table
    +-- feedback table
    +-- audit_log table
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- A Supabase Project (PostgreSQL)

### 2. Local Installation

```bash
# Clone the repository
git clone https://github.com/SumedhPatil1507/crediSense-ai.git
cd crediSense-ai

# Install all dependencies
pip install -r requirements.txt -r requirements-api.txt
```

### 3. Environment Setup
Rename `.env.example` to `.env` and configure your keys:
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key
API_SECRET_KEY=your-strong-random-key
```

### 4. Running the Application

**Run the Streamlit Client Dashboard:**
```bash
streamlit run app/app.py
```

**Run the FastAPI Backend:**
```bash
uvicorn api.main:app --reload --port 8000
```
*(Swagger UI available at `http://localhost:8000/docs`)*

**Run entirely with Docker Compose:**
```bash
docker-compose up --build
```

---

## 🛠️ Developer Workflow & GitHub Deployment

To update your code on GitHub and redeploy to Streamlit Cloud:

```bash
# Stage your changes
git add .

# Commit changes
git commit -m "feat: Integrated Macro Stress Testing and Agentic HITL"

# Push to your repository
git push origin main
```

**Streamlit Cloud Auto-Deployment:**
If you have connected this repository to Streamlit Community Cloud, it will automatically detect the push to the `main` branch and redeploy the app.

---

## 📊 Business Value

- **Macro Resilience**: Ensure your credit portfolio survives economic downturns through rigorous stress testing.
- **Regulatory Safety**: Prove to regulators that your models are not systematically biased using the Fairness Audit dashboard.
- **Analyst Efficiency**: Agentic HITL drastically reduces manual review time by pre-computing rationale.
- **Zero-Downtime MLOps**: Self-healing triggers ensure the model adapts to data drift without manual intervention.

---

## 📚 Citations

- LightGBM: https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
- SHAP: https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
- Fairlearn: https://fairlearn.org/
- World Bank Macro Data: https://data.worldbank.org
