# Financial Fraud Detection System

A risk-tiered financial fraud detection pipeline built with XGBoost and SHAP — not just a binary flag, but a scored, explainable decision per transaction.

## Risk Tiers

| Tier | Score | Decision |
|---|---|---|
| 🟢 Low Risk | 0 – 40 | Auto-approve |
| 🟡 Medium Risk | 41 – 70 | Flag for human review |
| 🔴 High Risk | 71 – 100 | Auto-block |

## Key Design Decisions

**Why split before SMOTE?** Prevents data leakage — synthetic fraud examples from training must never appear in the test set.

**Why sampling_strategy=0.10?** 50/50 SMOTE creates an artificial world. 10% improves signal while staying realistic.

**Why precision-recall AUC over ROC-AUC?** ROC-AUC inflates scores on imbalanced data. PR-AUC focuses on the minority class.

**Why F-beta with beta=0.5?** False positives have a business cost. Beta < 1 penalises them more than missed fraud.

## Tech Stack

Python · XGBoost · imbalanced-learn · SHAP · scikit-learn · matplotlib · seaborn

## Setup

```bash
pip install -r requirements.txt
# Download creditcard.csv from kaggle.com/datasets/mlg-ulb/creditcardfraud
python fraud_detection.py
```

## Dataset

[Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) · 284,807 transactions · 492 fraud cases

## Author

**Saaswati** · B.Tech CSE (AI/ML) · Lovely Professional University · [saaswati.github.io](https://saaswati.github.io)
