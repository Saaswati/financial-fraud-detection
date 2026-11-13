# Financial Fraud Detection System

A risk-tiered financial fraud detection pipeline built with XGBoost and SHAP — not just a binary flag, but a scored, explainable decision per transaction.

## Risk Tiers

| Tier | Score | Decision |
|---|---|---|
| 🟢 Low Risk | 0 – 40 | Auto-approve |
| 🟡 Medium Risk | 41 – 70 | Flag for human review |
| 🔴 High Risk | 71 – 100 | Auto-block |

## Results (full dataset, held-out test set of 56,962 transactions / 98 frauds)

| Metric | Value |
|---|---|
| PR-AUC (average precision) | **0.853** (random baseline = 0.0017) |
| ROC-AUC (for reference only) | 0.976 |
| Fraud precision | **0.929** — 79 of 85 flagged transactions were real fraud |
| Fraud recall | **0.806** — 79 of 98 frauds caught |
| F0.5 | **0.902** |
| Decision threshold | 0.81 (chosen on validation to maximise F0.5) |

Risk tiers on the test set: 56,874 low (17 fraud) · 1 medium (0 fraud) · 87 high (81 fraud). Only 6 legitimate customers out of 56,864 were auto-blocked.

## Key Design Decisions

**Why a train / validation / test split, done first?** Everything that learns from data (scaler, SMOTE, model, calibrator, threshold) is fitted on train or validation only. The test set is used once, at the end. This prevents data leakage.

**Why split before SMOTE?** Synthetic fraud examples are interpolations of real ones — if SMOTE ran before the split, near-copies of test frauds would sit in training.

**Why sampling_strategy=0.10?** 50/50 SMOTE creates an artificial world. 10% improves signal while staying realistic.

**Why no `scale_pos_weight`?** Stacked on SMOTE it would re-weight fraud ~10× — effectively 50/50 again, and overconfident probabilities. One imbalance fix, not two.

**Why calibrate probabilities?** The model trains on ~9% fraud but reality is ~0.17%. Isotonic regression (fitted on validation) maps raw scores to real fraud frequencies, so the 0–100 risk score means what it says.

**Why tune the threshold?** `predict()` silently uses 0.5. The threshold is chosen on validation to maximise F0.5.

**Why precision-recall AUC over ROC-AUC?** ROC-AUC inflates scores on imbalanced data (huge true-negative count keeps the false-positive rate tiny). PR-AUC focuses on the minority class.

**Why F-beta with beta=0.5?** False positives have a business cost. Beta < 1 weights precision more than recall. The medium tier routes uncertain cases to human review instead of forcing a block/approve decision.

## Limitations

- V1–V28 are anonymised PCA components, so SHAP can say *which* component drove a decision but not its business meaning.
- The split is random, not time-based; the dataset covers only 2 days, so concept drift can't be measured.
- The model separates classes so sharply that very few transactions land in the medium tier; tier boundaries should be set from review-team capacity and fraud-loss costs.

## Tech Stack

Python · XGBoost · imbalanced-learn · SHAP · scikit-learn · matplotlib · seaborn

## Setup

```bash
pip install -r requirements.txt
# Download creditcard.csv from kaggle.com/datasets/mlg-ulb/creditcardfraud into this folder
python fraud_detection.py
# or point to it elsewhere: CREDITCARD_CSV=/path/to/creditcard.csv python fraud_detection.py
```

## Dataset

[Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) · European cardholders, Sept 2013 (2 days) · 284,807 transactions · 492 fraud cases (0.17%) · amounts in euros

## Author

**Saaswati** · B.Tech CSE (AI/ML) · Lovely Professional University · [saaswati.github.io](https://saaswati.github.io)
