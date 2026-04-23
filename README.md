# CrediSense AI

An AI-powered **Credit Risk Scoring System** that predicts loan default probability with explainability, live macroeconomic context, real-time news, confidence intervals, adverse action notices, PDF reports, and analyst feedback loops.

**Live App:** https://credisense-ai-uzzvdsxuuxdbocfwxhmqcc.streamlit.app/

![CI](https://github.com/SumedhPatil1507/crediSense-ai/actions/workflows/ci.yml/badge.svg)

---

## Features

| Page | What it does |
|------|-------------|
| EDA | Violin plots, correlation heatmap, geographic treemap, profession analysis, interactive deep dive, live macro dashboard, RSS news feed |
| Model | Prediction + 95% bootstrap CI + risk gauge + what-if simulator + ROC/PR/calibration curves + model comparison + threshold analysis |
| Explainability | SHAP summary, waterfall, dependence plots + LIME local explanations |
| Chatbot | Risk prediction with rule-based explanation + credit risk Q&A knowledge base |
| Logs | Usage logs, analyst feedback, audit trail (PII-hashed), cost-benefit tracker, performance drift monitor |

---

## Tech Stack

- **ML:** LightGBM, Scikit-learn (Pipeline + ColumnTransformer)
- **Explainability:** SHAP TreeExplainer + LIME TabularExplainer
- **Live Data:** World Bank Open Data API, RSS news feeds (feedparser)
- **PDF Reports:** ReportLab (full Unicode support)
- **Frontend:** Streamlit + Plotly
- **Compliance:** ECOA/FCRA adverse action notices, audit logs, PII masking
- **CI:** GitHub Actions

---

## Model

**Algorithm:** LightGBM (LGBMClassifier)

Chosen over Random Forest and XGBoost because:
- Handles class imbalance natively via `class_weight='balanced'`
- Faster training on large tabular datasets (gradient-based leaf-wise splitting)
- Better performance on high-cardinality categorical features (Profession, City, State)

**Hyperparameters (tuned via RandomizedSearchCV):**

| Parameter | Value |
|-----------|-------|
| n_estimators | 200 |
| max_depth | 8 |
| learning_rate | 0.05 |
| colsample_bytree | 0.8 |
| subsample | 0.8 |
| class_weight | balanced |

**Performance (20% stratified holdout):**

| Metric | Score |
|--------|-------|
| ROC-AUC | ~0.97 |
| Gini Coefficient | ~0.94 |
| KS Statistic | ~0.75 |
| PR-AUC | ~0.72 |
| F1 (Risk class) | ~0.72 |
| Brier Score | ~0.08 |

---

## Project Structure

```
credisense-ai/
├── app/
│   ├── app.py                  # Landing page
│   └── pages/
│       ├── 1_EDA.py            # Data intelligence hub
│       ├── 2_Model.py          # Prediction + evaluation + what-if
│       ├── 3_Explainability.py # SHAP + LIME
│       ├── 4_Chatbot.py        # Risk assistant + Q&A
│       └── 5_Logs.py           # Logs + audit + cost-benefit
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── encoding.py
│   ├── pipeline.py
│   ├── train.py
│   ├── evaluate.py             # AUC, Gini, KS, PR-AUC, calibration
│   ├── tuning.py
│   ├── explainability.py       # SHAP helpers
│   ├── confidence_intervals.py # Bootstrap CI
│   ├── adverse_action.py       # ECOA/FCRA notices
│   ├── feedback.py             # Usage + feedback logging
│   ├── validation.py           # Input validation, PII masking, audit log
│   ├── live_data.py            # World Bank API + RSS news scraper
│   └── report.py               # PDF report generator (ReportLab)
├── models/
│   ├── model.pkl
│   └── columns.json
├── data/
│   ├── loan_cleaned.csv        # 252k loan records
│   └── sample_input.csv
├── utils.py
├── requirements.txt
└── .github/workflows/ci.yml
```

---

## Business Impact

### Problem Statement
Indian banks and NBFCs lose billions annually to loan defaults. Manual credit assessment is slow, inconsistent, and unscalable. CrediSense AI automates and augments this process with a production-grade ML system.

### Quantified Impact

| Impact Area | Without Model | With CrediSense AI |
|-------------|--------------|-------------------|
| Default detection | ~50% (random) | ~85% recall at 0.5 threshold |
| Review time per application | 2-4 hours (manual) | < 1 second |
| False approval rate | High | Reduced by ~70% vs baseline |
| Regulatory compliance | Manual documentation | Auto-generated adverse action notices |
| Analyst productivity | 1 decision/hour | 100s of decisions/hour with oversight |

### Financial Model (illustrative, 1000 applications/month)
- Avg loan: Rs 5,00,000 | Default rate: 12% | Loss given default: 60%
- **Without model:** Expected monthly loss = Rs 3.6 Cr (120 defaults x Rs 3L loss each)
- **With model (85% recall):** Catches ~102 of 120 defaults
- **Net loss reduction:** ~Rs 3.06 Cr/month
- **Annual savings estimate:** ~Rs 36 Cr on a 1000-application/month portfolio

### Decision Framework

| Risk Score | Decision | Business Rationale |
|-----------|----------|--------------------|
| < 30% | Approve | Low default probability, revenue opportunity |
| 30-60% | Manual Review | Borderline — human judgment required |
| > 60% | Reject | High default risk, capital preservation |

### Regulatory Alignment
- **ECOA (Equal Credit Opportunity Act):** Adverse action notices generated for every rejection with specific reasons
- **FCRA (Fair Credit Reporting Act):** Applicants informed of their right to know reasons for denial
- **Basel II Expected Loss Framework:** Cost-benefit tracker uses EL = PD x LGD x EAD methodology
- **RBI Guidelines:** Model decisions contextualized with live RBI repo rate and NPA data

### Stakeholder Value

| Stakeholder | Value Delivered |
|-------------|----------------|
| Loan Officers | Instant risk score + explanation, reducing cognitive load |
| Risk Managers | Threshold tuning, model comparison, drift monitoring |
| Compliance Teams | Audit trail, PII-masked logs, adverse action notices |
| Business Leaders | Cost-benefit P&L simulation, approval rate analytics |
| Data Scientists | SHAP/LIME explainability, feedback loop for retraining |

---

## Citations

- Dataset: https://www.kaggle.com/datasets/subhamjain/loan-prediction-based-on-customer-behavior
- LightGBM: https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
- SHAP: https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
- LIME: https://arxiv.org/abs/1602.04938
- Metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
- Compliance: https://www.bis.org/publ/bcbs128.htm
- Live Data: https://data.worldbank.org

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

---

## Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to streamlit.io/cloud -> New app
3. Set main file: `app/app.py`
4. Click Deploy

```bash
git add .
git commit -m "your message"
git push
```
