# CrediSense risk model card

## Intended use

Exploratory credit-risk scoring demonstration. A score is not, by itself, a
credit decision or a substitute for a lender's validated underwriting policy.

## Model and data

- Implementation: scikit-learn pipeline with LightGBM classifier.
- Training data and preparation: see `DATASET_ANALYSIS.md` and `src/train.py`.
- Current model artifact: `models/model.pkl`; record its SHA-256 and the source
  dataset version for every evaluated release.
- Features: see `models/columns.json` and `src/feature_engineering.py`.

## Evaluation required before use

Publish a dated report with temporal or otherwise leakage-safe split details,
sample counts, class prevalence, ROC-AUC, PR-AUC, Brier score, calibration,
threshold-specific precision/recall, confidence intervals, and subgroup
fairness results. Include comparison to a simple baseline and document missing
labels, selection effects, and data limitations. Do not treat headline metrics
in screenshots or README prose as production validation.

## Limitations and monitoring

Monitor feature freshness, missingness, score distribution, calibration, and
outcomes where labels become available. Re-evaluate for population changes,
policy changes, and subgroup performance. Define an owner, alert thresholds,
rollback artifact, and review cadence before deployment. No production approval
or regulatory certification is implied by this repository.
