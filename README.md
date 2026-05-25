# CrediSense AI

A production-grade, end-to-end **Credit Risk Scoring System** built to enterprise standards. Predicts loan default probability with full MLOps infrastructure, regulatory compliance, live data, and fairness auditing.

**Live App:** https://credisense-ai-uzzvdsxuuxdbocfwxhmqcc.streamlit.app/

![CI](https://github.com/SumedhPatil1507/crediSense-ai/actions/workflows/ci.yml/badge.svg)

---

## What Makes This Production-Grade

| Capability | Implementation |
|---|---|
| ML Model | LightGBM, hyperparameter tuning, class imbalance handling |
| Explainability | SHAP beeswarm, waterfall, dependence, interaction heatmap |
| Fairness Audit | Subgroup metrics, Disparate Impact Ratio (4/5ths rule), ECOA compliance |
| Confidence Intervals | Bootstrap 95% CI on every prediction |
| Regulatory Compliance | ECOA/FCRA adverse action notices, audit trail, PII masking |
| Database | SQLite (zero-config, persistent) |
| REST API | FastAPI with Pydantic v2 validation, API key auth, Swagger docs |
| Batch Processing | Up to 500 applicants per API call |
| Model Versioning | Registry with MD5 hash verification and performance tracking |
| Drift Monitoring | PSI (score-level) + CSI (feature-level) with auto-alerts |
| Human-in-the-Loop | Auto-queue borderline cases for analyst review |
| Shadow Mode | Run challenger model silently alongside champion |
| Stress Testing | Scenario-based stress tests (recession, rate hike, income shock) |
| Alerting | Webhook (Slack/Teams/Discord) + email alerts |
| Containerization | Docker + docker-compose |
| CI/CD | GitHub Actions with pytest unit + integration tests |
| Live Data | World Bank API + RBI repo rate + RSS news feeds (no API key) |
| PDF Reports | ReportLab reports with CI, explanation, adverse notice |

---

## Pages

| Page | What it does |
|------|-------------|
| Stress Testing | Scenario analysis — recession, rate hike, income shock impact on portfolio |
| Model | Predict with 95% CI + what-if simulator + ROC/PR/calibration + model comparison + threshold analysis + live macro context |
| Explainability | SHAP beeswarm + waterfall + dependence + interaction heatmap + fairness audit |
| Chatbot | Risk assistant with real inputs (LPA, age, years) + credit risk Q&A |
| Logs | Usage logs, feedback, audit trail, cost-benefit tracker, drift monitor |
| Operations | HITL queue, PSI/CSI drift, shadow mode, model registry, alert setup |

---

## Model Performance

| Metric | Score | Industry Benchmark |
|--------|-------|-------------------|
| ROC-AUC | ~0.97 | > 0.75 = good |
| Gini Coefficient | ~0.94 | > 0.60 = good |
| KS Statistic | ~0.75 | > 0.40 = good |
| PR-AUC | ~0.72 | Imbalanced baseline ~0.12 |
| F1 (Risk class) | ~0.72 | Threshold-dependent |
| Brier Score | ~0.08 | Lower = better calibrated |

**Algorithm:** LightGBM — gradient-based leaf-wise splitting, native class imbalance handling, superior performance on high-cardinality categoricals (Profession, City, State).

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
| Regulatory documentation | Manual | Auto-generated (ECOA/FCRA) |

### Financial Model
- Avg loan: Rs 5,00,000 | Default rate: 12% | LGD: 60%
- Without model: Expected monthly loss = Rs 3.6 Cr
- With model (85% recall): Catches ~102 of 120 defaults
- Net loss reduction: ~Rs 3.06 Cr/month = Rs 36 Cr/year on 1000 apps/month

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
- 4/5ths Rule: Disparate Impact Ratio fairness audit
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
│       ├── 1_Stress_Testing.py  # Scenario stress tests
│       ├── 2_Model.py           # Predict + CI + what-if + evaluation + live macro
│       ├── 3_Explainability.py  # SHAP + fairness audit
│       ├── 4_Chatbot.py         # Risk assistant + Q&A
│       ├── 5_Logs.py            # Logs + audit + cost-benefit
│       └── 6_Operations.py      # HITL + drift + shadow + registry + alerts
├── src/
│   ├── config.py            # Env-based config (no hardcoded secrets)
│   ├── database.py          # SQLite persistence layer
│   ├── schemas.py           # Pydantic v2 strict validation schemas
│   ├── model_registry.py    # Version tracking + hash verification
│   ├── drift_monitor.py     # PSI + CSI computation
│   ├── hitl_queue.py        # Human-in-the-loop queue
│   ├── shadow_mode.py       # Champion vs challenger comparison
│   ├── alerts.py            # Webhook + email alerting
│   ├── adverse_action.py    # ECOA/FCRA notices
│   ├── confidence_intervals.py  # Bootstrap CI
│   ├── fairness.py          # Subgroup metrics + Disparate Impact Ratio
│   ├── report.py            # PDF report generator (ReportLab)
│   ├── stress_test.py       # Scenario stress testing
│   ├── evaluate.py          # AUC, Gini, KS, PR-AUC, calibration
│   ├── explainability.py    # SHAP helpers
│   ├── validation.py        # Input validation + PII masking + audit
│   └── live_data.py         # World Bank API + RSS news scraper
├── tests/
│   ├── test_pipeline.py     # Unit tests
│   └── test_api.py          # API integration tests
├── db/
│   └── schema.sql           # Reference SQL schema
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
pip install -r requirements.txt -r requirements-api.txt
streamlit run app/app.py

# FastAPI backend
uvicorn api.main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs

# Docker
docker-compose up
```

---

## API Usage

```bash
# Single prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "x-api-key: dev-secret-change-in-prod" \
  -H "Content-Type: application/json" \
  -d '{"income_lpa": 10, "age_years": 30, "experience_years": 5}'

# Batch (up to 500)
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "x-api-key: dev-secret-change-in-prod" \
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
- Basel II: https://www.bis.org/publ/bcbs128.htm
- World Bank: https://data.worldbank.org
- scikit-learn: https://scikit-learn.org/stable/modules/model_evaluation.html
- ECOA: https://www.consumerfinance.gov/consumer-tools/credit-reports-and-scores/
