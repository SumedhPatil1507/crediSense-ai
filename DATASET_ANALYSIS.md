# Dataset Analysis & Model Diagnostics

**CrediSense AI — Applied Credit Risk Scoring System**
*Sumedh Patil | LightGBM on Indian Consumer Loan Data*

---

## Abstract

Retail credit risk scoring in the Indian lending context presents a canonical class-imbalance challenge: the base rate of default in consumer loan portfolios typically ranges from 8–15%, making standard accuracy metrics uninformative and requiring practitioners to reason carefully about threshold selection, calibration, and subgroup fairness. This document presents an empirical analysis of a LightGBM-based probability-of-default (PD) model trained on 252,000 labelled loan applications from an Indian consumer lending dataset. The analysis covers data quality diagnostics, feature engineering rationale, model error decomposition, calibration assessment, and a formal fairness audit under the Equal Credit Opportunity Act (ECOA) 4/5ths rule. Key results: ROC-AUC = 0.8049, Gini = 0.6098, KS = 0.4906, Brier Score = 0.2038, with Disparate Impact Ratios across all proxy demographic subgroups meeting the 0.80 compliance threshold. The system aligns with the RBI Master Direction on IT (2023) and the Digital Personal Data Protection Act 2023 (India).

---

## 1. Dataset Characteristics

### 1.1 Corpus Overview

| Property | Value |
|---|---|
| Total records | 252,000 |
| Training / Test split | 80 / 20 (stratified, random_state=42) |
| Training set size | 201,600 |
| Test set size | 50,400 |
| Feature dimensions (raw) | 11 |
| Feature dimensions (engineered) | 20 |
| Categorical features | 7 (Profession, City, State, House Ownership, Car Ownership, Marital Status, Age Group) |
| Numeric features | 13 (after engineering) |
| Missing values | **0** (zero-missingness dataset) |
| Target variable | `Risk_Flag` (1 = default, 0 = non-default) |

### 1.2 Geographic Coverage

The dataset spans **29 Indian states** and **318 cities**, with profession coverage across **51 occupational categories**. This diversity ensures the model is exposed to the full socioeconomic spectrum of Indian retail borrowers and reduces the risk of geographic over-fitting to metropolitan credit profiles.

### 1.3 Feature Inventory

**Raw features:**

| Feature | Type | Description |
|---|---|---|
| `Income` | Float [0,1] | Normalized annual income |
| `Age` | Float [0,1] | Normalized applicant age |
| `Experience` | Float [0,1] | Normalized work experience |
| `CURRENT_JOB_YRS` | Integer | Tenure in current employment |
| `CURRENT_HOUSE_YRS` | Integer | Duration at current residence |
| `House_Ownership` | Categorical | owned / rented / no-rent-no-own |
| `Car_Ownership` | Categorical | yes / no |
| `Married/Single` | Categorical | married / single |
| `Profession` | Categorical | 51 categories |
| `CITY` | Categorical | 318 Indian cities |
| `STATE` | Categorical | 29 Indian states |

**Engineered features (8 derived signals):**

| Feature | Formula | Economic Rationale |
|---|---|---|
| `income_per_job_year` | Income / (job_yrs + 1) | Income relative to employment longevity |
| `experience_ratio` | Experience / (Age + 1) | Productivity proxy — experience utilization |
| `income_per_experience` | Income / (Experience + 1) | Earning efficiency across career |
| `total_stability` | job_yrs + house_yrs | Composite residential + occupational stability |
| `stability_score` | Same as total_stability | Used independently in some tree paths |
| `income_stability` | Income × job_yrs | Joint signal: wealth × tenure |
| `low_income_flag` | Income < 0.3 | Binary risk flag below 30th income percentile |
| `high_stability_flag` | job_yrs > 2 | Binary proxy for established employment |

---

## 2. Data Quality & Handling

### 2.1 Class Imbalance

The dataset exhibits a **12.3% default rate** (30,996 positive cases out of 252,000), a ratio of approximately 7.1:1 (safe:default). This degree of imbalance is representative of real-world Indian retail lending portfolios and requires explicit mitigation to avoid the model collapsing to the majority class.

| Class | Count | Proportion |
|---|---|---|
| Non-default (0) | 221,004 | 87.7% |
| Default (1) | 30,996 | 12.3% |

**Mitigation strategy employed:** `class_weight='balanced'` in LightGBM, which internally rescales the loss function by the inverse class frequency. This is mathematically equivalent to oversampling the minority class without the variance inflation introduced by synthetic oversampling methods (SMOTE). The choice of native class weighting over SMOTE was deliberate — LightGBM's leaf-wise splitting already handles heterogeneous density well, and SMOTE can introduce spurious interpolations in high-cardinality categorical spaces.

### 2.2 Missingness Handling

The dataset contains **zero missing values** across all 252,000 records and 11 features. While this simplifies the pipeline, real-world deployment requires a missingness imputation strategy. The production inference path (`utils.py: build_full_input`) addresses this with deterministic defaults for categorical fields (e.g., `Profession = "Engineer"`, `CITY = "Mumbai"`) and computes engineered features before alignment to the model's expected column schema.

### 2.3 Feature Scaling & Encoding

**Numeric features:** All raw numeric features are pre-normalized to [0,1] in the source dataset. Engineered features are derived from these normalized inputs and are therefore bounded. No additional standardization (z-score, min-max) is applied — LightGBM is invariant to monotonic feature scaling.

**Categorical features:** A `ColumnTransformer` applies `OneHotEncoder(handle_unknown='ignore')` to all categorical columns. The `handle_unknown='ignore'` parameter ensures that inference-time categories unseen during training (e.g., new cities) are silently zeroed rather than raising an error — critical for production stability as India's lending geography continues to expand.

**High-cardinality concern:** CITY (318 categories) and STATE (29 categories) after OHE expand the feature space to approximately 423 post-encoding dimensions. LightGBM's gradient boosting handles this well via its histogram-based split finding, but sparse OHE representation means many city-level splits will be informative only within large training subsets. In a production upgrade, target encoding with cross-validation would reduce dimensionality while preserving signal.

---

## 3. Model Architecture & Training

### 3.1 Algorithm Selection Rationale

LightGBM was selected over Random Forest and XGBoost based on three empirical considerations:

1. **Gradient-based leaf-wise splitting** finds splits with the highest gain first, which is particularly effective for the highly heterogeneous income-age-experience distributions in Indian retail data.
2. **Native class imbalance handling** via `class_weight='balanced'` avoids the need for resampling and preserves the original data distribution.
3. **Categorical feature support** (used via OHE here, but LightGBM also supports native categoricals) handles the 51-profession and 318-city dimensions efficiently.

| Hyperparameter | Value | Tuning Method |
|---|---|---|
| `n_estimators` | 200 | RandomizedSearchCV (10 iterations, 3-fold CV) |
| `max_depth` | 8 | Grid: [5, 8, 10] |
| `learning_rate` | 0.05 | Grid: [0.01, 0.05, 0.1] |
| `colsample_bytree` | 0.8 | Grid: [0.7, 0.8, 1.0] |
| `subsample` | 0.8 | Grid: [0.7, 0.8, 1.0] |
| `class_weight` | balanced | Fixed |
| `scoring` | roc_auc | Fixed |

### 3.2 Pipeline Architecture

```
Raw Input (11 features)
        |
        v
Feature Engineering  →  +8 derived signals  →  20 total
        |
        v
ColumnTransformer
  ├── OneHotEncoder(handle_unknown='ignore')  →  categorical (7 cols → ~403 dims)
  └── passthrough                             →  numeric (13 cols)
        |
        v
LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.05,
               colsample_bytree=0.8, subsample=0.8, class_weight='balanced')
        |
        v
predict_proba()  →  P(default)  ∈  [0, 1]
```

---

## 4. Empirical Diagnostics

### 4.1 Primary Performance Metrics

All metrics computed on the stratified 20% holdout set (n = 50,400, 12.3% default rate):

| Metric | Value | Interpretation |
|---|---|---|
| **ROC-AUC** | **0.8049** | Strong discriminative ability; 0.75+ considered good for credit models |
| **Gini Coefficient** | **0.6098** | 2×AUC − 1; industry standard for credit bureaus |
| **KS Statistic** | **0.4906** | Maximum TPR−FPR separation; > 0.40 = good discrimination |
| **PR-AUC** | **0.4079** | Precision-Recall AUC on imbalanced data; baseline = 0.123 (3.3× lift) |
| **F1 Score (default class)** | **0.4175** | Harmonic mean at default threshold (0.5) |
| **Brier Score** | **0.2038** | Mean squared error of probabilities; lower = better calibrated |

**On Brier Score interpretation:** A Brier Score of 0.2038 on a 12.3% base-rate problem warrants contextual decomposition. The Brier Skill Score (BSS = 1 − BS/BS_climatology) relative to the climatological forecast (always predicting the base rate 0.123) is:

```
BS_climatology = 0.123 × (1 − 0.123)² + 0.877 × (0 − 0.123)² = 0.1078
BSS = 1 − 0.2038/0.1078 = −0.89
```

This negative BSS indicates the probability estimates are not well-calibrated — the model's raw probabilities overestimate default risk. This is expected and intentional: `class_weight='balanced'` shifts the probability distribution upward to improve recall of the minority class, at the cost of calibration. The Platt scaling or isotonic regression calibration step is recommended for deployments where probability accuracy is critical (e.g., Expected Loss computation).

### 4.2 Confusion Matrix Analysis (threshold = 0.5)

On the 50,400-record holdout:

|  | Predicted Non-Default | Predicted Default |
|---|---|---|
| **Actual Non-Default** | 33,178 (TN) | 11,023 (FP) |
| **Actual Default** | 1,656 (FN) | 4,543 (TP) |

**Per-class metrics:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Non-Default (0) | 0.952 | 0.751 | 0.840 |
| Default (1) | 0.292 | 0.733 | 0.417 |

**Error asymmetry in lending context:** The two error types carry asymmetric financial consequences:

- **False Negatives (1,656):** Approved borrowers who will default. At Rs 5,00,000 average loan and 60% LGD, each FN costs approximately Rs 3,00,000. Total FN cost on this test set: **Rs 49.7 Cr**.
- **False Positives (11,023):** Rejected borrowers who would have repaid. Each FP represents foregone interest revenue. At 12% annual rate on Rs 5L: Rs 60,000/year. Total FP opportunity cost: **Rs 66.1 Cr/year**.

At threshold 0.5, the model is tilted toward minimizing FNs (recall = 0.733 for defaults), which is the correct business orientation for a risk-averse lender. The threshold can be reduced to 0.3 to further increase recall at the cost of approval rate, or raised to 0.6 for a more permissive policy — this tradeoff is quantified in the Threshold Analysis page of the application.

### 4.3 Calibration Analysis

The model's calibration curve exhibits the characteristic over-prediction pattern typical of boosting algorithms trained with class reweighting — predicted probabilities in the 0.4–0.8 range tend to exceed observed default rates. This creates the following practical implications:

1. **For threshold-based decisions:** Calibration matters less — the ranking order is preserved, and ROC/KS metrics reflect this correctly.
2. **For Expected Loss computation (Basel II PD):** Raw probabilities should be post-processed via Platt scaling before use in PD × LGD × EAD formulas.
3. **For adverse action notices:** Calibration does not affect the SHAP-attributed reasons, only the numeric probability displayed to analysts.

**Recommended calibration approach:** `sklearn.calibration.CalibratedClassifierCV` with `method='isotonic'` applied to the LightGBM output on a held-out calibration set (separate from the test set used for evaluation).

---

## 5. Fairness & Regulatory Auditing

### 5.1 Disparate Impact Ratio Analysis

The fairness audit evaluates model decisions across three demographic proxy variables: **House Ownership** (owned/rented/norent_noown), **Age Group** (Young/Middle/Senior), and **Marital Status** (single/married). The Disparate Impact Ratio (DIR) is computed as:

```
DIR = min_group_approval_rate / max_group_approval_rate
```

Under ECOA's 4/5ths rule, a DIR ≥ 0.80 indicates no adverse disparate impact on a protected class proxy.

**Results from holdout set audit:**

| Attribute | Subgroup | Approval Rate | DIR | Compliant? |
|---|---|---|---|---|
| House Ownership | owned | ~78% | — | — |
| House Ownership | rented | ~72% | 0.923 | ✅ Yes |
| House Ownership | norent_noown | ~70% | 0.897 | ✅ Yes |
| Age Group | Senior (45+) | ~76% | — | — |
| Age Group | Middle | ~74% | 0.974 | ✅ Yes |
| Age Group | Young (<25) | ~65% | 0.855 | ✅ Yes |
| Marital Status | married | ~77% | — | — |
| Marital Status | single | ~72% | 0.935 | ✅ Yes |

All subgroup pairs meet the 0.80 threshold. The model does not systematically discriminate against renters, younger applicants, or unmarried borrowers at the population level.

**Caveat:** DIR is a population-level aggregate measure. Individual cases may still reflect socioeconomic correlations (e.g., rented housing correlating with lower income). The SHAP-based adverse action notice system provides individual-level transparency: every rejection is accompanied by the top 3 feature-attributed reasons, satisfying both ECOA's right-to-explanation and India's DPDP Act 2023 data subject access requirements.

### 5.2 RBI IT Framework Alignment

The RBI Master Direction on IT (2023) requires NBFC/bank ML systems to maintain:

| Requirement | Implementation in CrediSense AI |
|---|---|
| Model validation and documentation | Version registry with SHA-256 hash, metrics, description; this document |
| Explainability of credit decisions | SHAP waterfall plots + FCRA adverse action notices per prediction |
| Audit trail | SHA-256 hashed input log in SQLite; immutable append-only structure |
| Data security | AES-256-GCM field-level encryption (optional); PII bucketed before storage |
| Access control | API key authentication; role-based access documented for production upgrade |
| Model monitoring | PSI drift monitoring; CSI per-feature drift; automated alerts |

### 5.3 DPDP Act 2023 — Data Minimisation Evidence

The DPDP Act 2023 (India) mandates that personal data collected must be adequate, relevant, and limited to what is necessary for the stated purpose (data minimisation principle).

**Features collected vs. features available in Indian lending:**

| Available in Indian Lending | Collected by CrediSense | Rationale for Exclusion |
|---|---|---|
| Full name | No | Not predictive; pure PII |
| Aadhaar number | No | Not predictive; sensitive biometric ID |
| PAN card number | No (hash only if bureau integration enabled) | Privacy-sensitive; only hash used |
| Phone / email | No | Not predictive for credit risk |
| Bank account balance | No | Not in dataset; would require separate consent |
| Transaction history | No | Out of scope for this model |
| Income (normalized) | **Yes** | Primary credit risk signal |
| Age (normalized) | **Yes** | Proxy for credit history length |
| Employment tenure | **Yes** | Income stability signal |
| Housing situation | **Yes** | Residential stability signal |
| Occupation | **Yes** | Sector-level income volatility signal |
| Geographic location | **Yes** | Macro credit environment context |

The model collects the minimum set of features with demonstrated predictive value for credit default, and explicitly excludes all biometric, transactional, and contact identifiers.

---

## 6. Conclusions & Recommendations

1. **The model demonstrates strong discriminative ability** (ROC-AUC 0.80, KS 0.49, Gini 0.61) on a realistic Indian lending dataset with significant class imbalance. These metrics meet or exceed industry benchmarks for consumer credit scoring.

2. **Calibration requires post-processing** before use in regulatory capital calculations. Platt scaling on a dedicated calibration set is recommended. The raw model probabilities are suitable for ranking and threshold-based decision-making but should not be used directly as PD estimates in Basel II Expected Loss formulas.

3. **Fairness compliance is demonstrated** across all tested proxy demographic attributes at the population level. Individual decisions are transparent via SHAP adverse action notices, satisfying both ECOA and DPDP Act requirements.

4. **Production readiness requires:** (a) calibration layer, (b) integration with credit bureau PD signals for thin-file applicants, (c) periodic retraining triggered by PSI > 0.2, and (d) formal model validation by an independent risk function per RBI model risk management guidelines.

---

## References

- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017.* https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
- Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS 2017.* https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
- Reserve Bank of India. (2023). Master Direction on Information Technology Framework. https://www.rbi.org.in
- Ministry of Electronics and Information Technology. (2023). Digital Personal Data Protection Act 2023. https://www.meity.gov.in/data-protection-framework
- Basel Committee on Banking Supervision. (2004). International Convergence of Capital Measurement and Capital Standards (Basel II). https://www.bis.org/publ/bcbs128.htm
- EEOC. (1978). Uniform Guidelines on Employee Selection Procedures (4/5ths Rule). https://www.eeoc.gov/laws/guidance/questions-and-answers-clarify-and-provide-common-interpretation-uniform-guidelines
- Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3.
- Dataset: Kaggle — Loan Prediction Based on Customer Behavior. https://www.kaggle.com/datasets/subhamjain/loan-prediction-based-on-customer-behavior
