# CrediSense AI

A production-grade, end-to-end **Credit Risk Scoring System** built to enterprise standards. Predicts loan default probability with explainability, confidence intervals, regulatory compliance, live data, and full MLOps infrastructure.

**Live App:** https://credisense-ai-uzzvdsxuuxdbocfwxhmqcc.streamlit.app/

![CI](https://github.com/SumedhPatil1507/crediSense-ai/actions/workflows/ci.yml/badge.svg)

---

## What Makes This Production-Grade

| Capability | Implementation |
|---|---|
| ML Model | LightGBM with hyperparameter tuning, class imbalance handling |
| Explainability | SHAP (global + waterfall + dependence) + LIME (local) |
| Confidence Intervals | Bootstrap 95% CI on every prediction |
| Regulatory Compliance | ECOA/FCRA adverse action notices, audit trail, PII masking |
| Database | Supabase (PostgreSQL) with CSV fallback |
| REST API | FastAPI with Pydantic validation, API key auth, Swagger docs |
| Batch Processing | Up to 500 applicants per API call |
| Model Versioning | Registry with hash verification and performance tracking |
| Drift Monitoring | PSI (score-level) + CSI (feature-level) with auto-alerts |
| Human-in-the-Loop | Auto-queue borderline cases for analyst review |
| Shadow Mode | Run challenger model silently alongside champion |
| Alerting | Webhook (Slack/Teams/Discord) + email alerts |
| Containerization | Docker + docker-compose for API + Streamlit |
| CI/CD | GitHub Actions with pytest unit + integration tests |
| Live Data | World Bank API + RBI repo rate + RSS news feeds |
| PDF Reports | ReportLab reports with CI, explanation, adverse notice |

---

## Architecture

```
User/Client
    |
    v
FastAPI Backend (api/main.py)          Streamlit Dashboard (app/)
    |                                       |
    +-- /predict (single)                   +-- 1_EDA.py (data + live macro + news)
    +-- /predict/batch (up to 500)          +-- 2_Model.py (predict + evaluate + what-if)
    +-- /explain (SHAP top features)        +-- 3_Explainability.py (SHAP + LIME)
    +-- /feedback (analyst corrections)     +-- 4_Chatbot.py (risk assistant + Q&A)
    +-- /metrics (aggregate stats)          +-- 5_Logs.py (logs + audit + cost-benefit)
    +-- /health (system status)             +-- 6_Operations.py (HITL + drift + shadow)
    |
    v
Supabase (PostgreSQL) / CSV fallback
    |
    +-- predictions table
    +-- feedback table
    +-- audit_log table
```

---

## Model Performance

| Metric | Score | Industry Benchmark |
|--------|-------|-------------------|
| ROC-AUC | ~0.97 | > 0.75 = good |
| Gini Coefficient | ~0.94 | > 0.60 = good |
| KS Statistic | ~0.75 | > 0.40 = good |
| PR-AUC | ~0.72 | Imbalanced data baseline ~0.12 |
| F1 (Risk class) | ~0.72 | Depends on threshold |
| Brier Score | ~0.08 | Lower = better calibrated |

**Algorithm:** LightGBM — chosen for gradient-based leaf-wise splitting, native class imbalance handling, and superior performance on high-cardinality categorical features (Profession, City, State).

**Hyperparameters (RandomizedSearchCV):**

| Parameter | Value |
|-----------|-------|
| n_estimators | 200 |
| max_depth | 8 |
| learning_rate | 0.05 |
| colsample_bytree | 0.8 |
| subsample | 0.8 |
| class_weight | balanced |

---

## Business Impact

### Problem
Indian banks and NBFCs lose billions annually to loan defaults. Manual credit assessment is slow, inconsistent, and unscalable.

### Quantified Results (illustrative, 1000 applications/month)

| Metric | Without Model | With CrediSense AI |
|--------|--------------|-------------------|
| Default detection rate | ~50% (random) | ~85% recall |
| Review time per application | 2-4 hours | < 1 second |
| False approval rate | High | Reduced ~70% vs baseline |
| Regulatory documentation | Manual | Auto-generated |

### Financial Model
- Avg loan: Rs 5,00,000 | Default rate: 12% | LGD: 60%
- Without model: Expected monthly loss = Rs 3.6 Cr
- With model (85% recall): Catches ~102 of 120 defaults
- Net loss reduction: ~Rs 3.06 Cr/month = **Rs 36 Cr/year savings** on 1000 apps/month

### Decision Framework

| Risk Score | Decision | Rationale |
|-----------|----------|-----------|
| < 30% | Approve | Low default probability |
| 30-60% | Manual Review | Borderline — human judgment |
| > 60% | Reject | Capital preservation |

### Regulatory Alignment
- ECOA: Adverse action notices with specific decline reasons
- FCRA: Applicant right-to-know documentation
- Basel II: Expected Loss = PD x LGD x EAD methodology
- RBI IT Framework: Audit trail, access controls, data lineage

---

## Project Structure

```
credisense-ai/
├── api/
│   └── main.py              # FastAPI: predict, batch, explain, feedback, metrics
├── app/
│   ├── app.py               # Landing page with system status
│   └── pages/
│       ├── 1_EDA.py         # Data hub + live macro + news
│       ├── 2_Model.py       # Predict + CI + what-if + evaluation
│       ├── 3_Explainability.py  # SHAP + LIME
│       ├── 4_Chatbot.py     # Risk assistant + Q&A
│       ├── 5_Logs.py        # Logs + audit + cost-benefit
│       └── 6_Operations.py  # HITL + drift + shadow + registry + alerts
├── src/
│   ├── config.py            # Env-based config
│   ├── database.py          # Supabase + CSV dual backend
│   ├── model_registry.py    # Version tracking + hash verification
│   ├── drift_monitor.py     # PSI + CSI computation
│   ├── hitl_queue.py        # Human-in-the-loop queue
│   ├── shadow_mode.py       # Champion vs challenger comparison
│   ├── alerts.py            # Webhook + email alerting
│   ├── adverse_action.py    # ECOA/FCRA notices
│   ├── confidence_intervals.py  # Bootstrap CI
│   ├── report.py            # PDF report generator
│   ├── evaluate.py          # AUC, Gini, KS, PR-AUC, calibration
│   ├── explainability.py    # SHAP helpers
│   ├── validation.py        # Input validation + PII masking + audit
│   ├── live_data.py         # World Bank API + RSS scraper
│   └── feedback.py          # Legacy CSV logging
├── tests/
│   ├── test_pipeline.py     # Unit tests (validation, drift, adverse action, CI, HITL)
│   └── test_api.py          # API integration tests
├── db/
│   └── schema.sql           # Supabase PostgreSQL schema
├── models/
│   ├── model.pkl            # Trained LightGBM pipeline
│   ├── columns.json         # Expected feature columns
│   └── registry.json        # Model version registry
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── requirements-api.txt
```

---

## Run Locally

```bash
# Install dependencies
pip install -r requirements.txt -r requirements-api.txt

# Run Streamlit dashboard
streamlit run app/app.py

# Run FastAPI backend
uvicorn api.main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs

# Run both with Docker
docker-compose up
```

---

## Supabase Setup (for persistent storage)

1. Create free project at supabase.com
2. Run `db/schema.sql` in the SQL Editor
3. Add to environment or Streamlit secrets:
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key
API_SECRET_KEY=your-strong-random-key
```

---

## API Usage

```bash
# Single prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "x-api-key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"income_lpa": 10, "age_years": 30, "experience_years": 5}'

# Batch prediction
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "x-api-key: your-key" \
  -d '{"applicants": [{"income_lpa": 10, "age_years": 30, "experience_years": 5}]}'
```

---

## Deploy

```bash
git add .
git commit -m "your message"
git push
```

Streamlit Cloud auto-redeploys on push. For API, deploy to Railway/Render using the Dockerfile.

---

## Citations

- Dataset: https://www.kaggle.com/datasets/subhamjain/loan-prediction-based-on-customer-behavior
- LightGBM: https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
- SHAP: https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
- LIME: https://arxiv.org/abs/1602.04938
- Basel II: https://www.bis.org/publ/bcbs128.htm
- World Bank: https://data.worldbank.org
- scikit-learn: https://scikit-learn.org/stable/modules/model_evaluation.html
