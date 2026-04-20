# 💳 CrediSense AI

An AI-powered **Credit Risk Scoring System** that predicts loan default probability and provides explainable, business-ready financial decisions.

🌐 **Live App:** [https://credisense-ai-uzzvdsxuuxdbocfwxhmqcc.streamlit.app/](https://credisense-ai-uzzvdsxuuxdbocfwxhmqcc.streamlit.app/)

![CI](https://github.com/YOUR_USERNAME/credisense-ai/actions/workflows/ci.yml/badge.svg)

---

## 🚀 Features

| Page | Description |
|------|-------------|
| 📊 EDA | Interactive data analysis — risk distribution, income & experience breakdowns |
| 💳 Model | Live prediction with risk gauge, full model metrics, ROC curve, confusion matrix, threshold analysis |
| 🧠 Explainability | SHAP summary plot — feature-level explanation of predictions |
| 🤖 Chatbot | Natural language risk assistant — enter Income, Age, Experience to get instant prediction |

---

## 🛠️ Tech Stack

- **ML:** LightGBM, Scikit-learn (Pipeline + ColumnTransformer)
- **Explainability:** SHAP TreeExplainer (summary, waterfall, dependence plots)
- **LLM:** Groq API · llama3-8b-8192 (prediction explanations + conversational analyst)
- **Live Data:** World Bank Open Data API (GDP, unemployment, inflation, NPA ratio)
- **Frontend:** Streamlit + Plotly
- **Data:** Pandas, NumPy
- **CI:** GitHub Actions

---

## 🧠 Model

**Algorithm:** LightGBM (`LGBMClassifier`)

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

**Performance (20% holdout test set):**

| Metric | Score |
|--------|-------|
| ROC-AUC | ~0.97 |
| F1 Score (Risk class) | ~0.72 |

> Exact metrics vary by run. Upload `loan_cleaned.csv` in the Model page to compute live metrics.

---

## � Project Structure

```
credisense-ai/
├── app/
│   ├── app.py                  # Streamlit entry point
│   └── pages/
│       ├── 1_EDA.py            # Exploratory data analysis
│       ├── 2_Model.py          # Prediction + model evaluation
│       ├── 3_Explainability.py # SHAP explainability
│       └── 4_Chatbot.py        # AI risk assistant
├── src/
│   ├── config.py               # Paths and constants
│   ├── data_loader.py          # CSV loading
│   ├── preprocessing.py        # Data cleaning
│   ├── feature_engineering.py  # Feature creation
│   ├── encoding.py             # ColumnTransformer setup
│   ├── pipeline.py             # Sklearn pipeline
│   ├── train.py                # Training script
│   ├── evaluate.py             # Metrics + threshold analysis
│   └── tuning.py               # Hyperparameter search
├── models/
│   ├── model.pkl               # Trained pipeline
│   └── columns.json            # Expected feature columns
├── data/
│   ├── loan_cleaned.csv        # Cleaned dataset
│   └── sample_input.csv        # Sample input for explainability
├── utils.py                    # Input builder for inference
├── requirements.txt
└── .github/workflows/ci.yml    # GitHub Actions CI
```

---

## 💡 Business Value

- Automates loan approval decisions with 3-tier logic (Approve / Review / Reject)
- Threshold analysis lets risk teams tune precision vs recall tradeoff
- SHAP explanations make decisions auditable and regulation-friendly
- Reduces manual review overhead for financial institutions

---

## 📦 Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Add Groq API key (free at console.groq.com)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml and add your GROQ_API_KEY

# Start the app
streamlit run app/app.py
```

## 🔑 LLM Setup (Groq — Free)

1. Get a free API key at [https://console.groq.com](https://console.groq.com)
2. **Local:** copy `.streamlit/secrets.toml.example` → `.streamlit/secrets.toml` and add your key
3. **Streamlit Cloud:** go to App Settings → Secrets → add `GROQ_API_KEY = "gsk_..."`

The app works without the key (rule-based fallback), but LLM features require it.
