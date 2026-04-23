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

## Business Value

- 3-tier decision engine: Approve / Manual Review / Reject
- Bootstrap confidence intervals flag borderline cases for human review
- ECOA/FCRA-compliant adverse action notices for every rejection
- SHAP + LIME explanations make decisions auditable
- Analyst feedback loop captures corrections for future retraining
- Cost-benefit tracker simulates P&L impact of model decisions
- Live macro dashboard contextualises predictions with RBI/World Bank data
- Real-time RSS news feed for credit risk and banking developments

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
