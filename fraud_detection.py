# =============================================================================
# FINANCIAL FRAUD DETECTION SYSTEM
# Risk-Tiered Scoring Engine with SHAP Per-Transaction Explanations
#
# Dataset : Kaggle Credit Card Fraud Detection
#           kaggle.com/datasets/mlg-ulb/creditcardfraud
# Author  : Saaswati
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    fbeta_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
RANDOM_STATE     = 42
TEST_SIZE        = 0.20
SMOTE_RATIO      = 0.10   # After SMOTE: fraud = 10% of legitimate count
TIER_LOW_MAX     = 40     # 0–40  → low risk (auto-approve)
TIER_MED_MAX     = 70     # 41–70 → medium risk (human review)
                           # 71–100 → high risk (auto-block)
FBETA_BETA       = 0.5    # Beta < 1 weights precision over recall


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  UNIT 1 — LOADING DATA")
print("=" * 65)

df = pd.read_csv('creditcard.csv')

print(f"  Dataset shape : {df.shape}")
print(f"\n  Class distribution:")
print(df['Class'].value_counts().rename({0: 'Legitimate (0)', 1: 'Fraud (1)'}).to_string())
print(f"\n  Fraud rate    : {df['Class'].mean() * 100:.4f}%")
print(f"\n  → This is why accuracy is a useless metric here.")
print(f"    A model that always says 'not fraud' is {(1 - df['Class'].mean())*100:.2f}% accurate.")
print(f"    It's also completely useless.")


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT 2 — EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  UNIT 2 — EDA")
print("=" * 65)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Financial Fraud Detection — EDA', fontsize=14, fontweight='bold', y=1.01)

# 2a. Class imbalance
counts = df['Class'].value_counts()
bars = axes[0].bar(
    ['Legitimate', 'Fraud'], counts.values,
    color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=0.7
)
axes[0].set_title('Class Imbalance', fontweight='bold')
axes[0].set_ylabel('Transaction Count')
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 1000,
                 f'{val:,}', ha='center', fontweight='bold', fontsize=10)

# 2b. Amount distribution
axes[1].hist(df[df['Class']==0]['Amount'], bins=60, alpha=0.6,
             color='#2ecc71', label='Legitimate', density=True)
axes[1].hist(df[df['Class']==1]['Amount'], bins=60, alpha=0.6,
             color='#e74c3c', label='Fraud', density=True)
axes[1].set_title('Transaction Amount Distribution', fontweight='bold')
axes[1].set_xlabel('Amount (₹ / $)')
axes[1].set_xlim(0, 2500)
axes[1].legend()

# 2c. Time distribution (hours into the data collection window)
axes[2].hist(df[df['Class']==0]['Time']/3600, bins=48, alpha=0.6,
             color='#2ecc71', label='Legitimate', density=True)
axes[2].hist(df[df['Class']==1]['Time']/3600, bins=48, alpha=0.6,
             color='#e74c3c', label='Fraud', density=True)
axes[2].set_title('Transaction Timing\n(Hours from dataset start)', fontweight='bold')
axes[2].set_xlabel('Hours')
axes[2].legend()

plt.tight_layout()
plt.savefig('01_eda.png', dpi=150, bbox_inches='tight')
plt.show()
print("  → Saved: 01_eda.png")


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT 3 — PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  UNIT 3 — PREPROCESSING")
print("=" * 65)

# V1–V28 are already PCA-transformed (zero mean, unit variance).
# Amount and Time are raw — we scale them to the same range.
# If we don't scale them, their raw magnitude will dominate the model
# and distort SHAP values.

scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_scaled']   = scaler.fit_transform(df[['Time']])
df.drop(columns=['Amount', 'Time'], inplace=True)

FEATURES = [c for c in df.columns if c != 'Class']
X = df[FEATURES]
y = df['Class']

print(f"  Feature count : {len(FEATURES)}")
print(f"  → Scaled Amount and Time. V1–V28 untouched (already PCA-transformed).")


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT 4 — TRAIN-TEST SPLIT  ← MUST HAPPEN BEFORE SMOTE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  UNIT 4 — TRAIN / TEST SPLIT")
print("=" * 65)

# WHY SPLIT BEFORE SMOTE:
# SMOTE creates synthetic fraud examples by interpolating between real ones.
# If you SMOTE the full dataset first, synthetic points from training fraud
# examples bleed into the test set. Your test evaluation becomes optimistic —
# the model has essentially already seen variations of what it's being tested on.
# Real-world performance will be worse than what you measured. This is data leakage.

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y          # preserves the fraud ratio in both splits
)

print(f"  Train : {X_train.shape[0]:>7,} rows | Fraud: {y_train.sum():>4}")
print(f"  Test  : {X_test.shape[0]:>7,} rows | Fraud: {y_test.sum():>4}")
print(f"\n  ⚠  SMOTE will only touch training data. Test set = real transactions only.")


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT 5 — SMOTE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  UNIT 5 — SMOTE (Synthetic Minority Oversampling)")
print("=" * 65)

# WHY NOT 50/50 SMOTE:
# Upsampling fraud to 50% of all transactions is unrealistic — the model
# learns a world that doesn't exist. Setting sampling_strategy=0.1 means
# "after SMOTE, fraud count = 10% of non-fraud count." Still much better
# than 0.17%, but not an artificial world.

smote = SMOTE(sampling_strategy=SMOTE_RATIO, random_state=RANDOM_STATE, n_jobs=-1)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"  Before SMOTE → Fraud: {y_train.sum():,}  |  Legit: {(y_train==0).sum():,}")
print(f"  After SMOTE  → Fraud: {y_train_res.sum():,}  |  Legit: {(y_train_res==0).sum():,}")
print(f"  New fraud rate in training set: {y_train_res.mean()*100:.2f}%")
print(f"\n  → sampling_strategy=0.10 means fraud = 10% of legit, not 50%.")
print(f"    More realistic. Model generalises better.")


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT 6 — XGBOOST TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  UNIT 6 — XGBOOST TRAINING")
print("=" * 65)

# WHY XGBOOST OVER RANDOM FOREST / LOGISTIC REGRESSION:
# XGBoost builds trees sequentially, each correcting the errors of the last.
# It handles mixed feature scales well, converges faster than Random Forest
# on tabular data, and has native support for imbalanced datasets via
# scale_pos_weight. SHAP was actually designed around tree ensembles first.

# scale_pos_weight: even after SMOTE, we tell XGBoost about the remaining ratio.
# Belt + suspenders. If SMOTE ratio is 0.1, pos_weight ~ 10.
neg_count = (y_train_res == 0).sum()
pos_count = (y_train_res == 1).sum()
spw = round(neg_count / pos_count, 2)

print(f"  scale_pos_weight = {spw}  (neg/pos after SMOTE)")

model = xgb.XGBClassifier(
    n_estimators       = 300,
    max_depth          = 6,
    learning_rate      = 0.05,
    subsample          = 0.8,      # row subsampling per tree → prevents overfitting
    colsample_bytree   = 0.8,      # feature subsampling per tree → prevents overfitting
    scale_pos_weight   = spw,
    eval_metric        = 'aucpr',  # area under precision-recall curve — right for imbalanced data
    random_state       = RANDOM_STATE,
    n_jobs             = -1
)

model.fit(
    X_train_res, y_train_res,
    eval_set=[(X_test, y_test)],
    verbose=50
)

print("\n  ✓ Training complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT 7 — EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  UNIT 7 — EVALUATION")
print("=" * 65)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

ap    = average_precision_score(y_test, y_proba)
fb    = fbeta_score(y_test, y_pred, beta=FBETA_BETA)

print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
print(f"  Average Precision (PR-AUC) : {ap:.4f}")
print(f"  F-{FBETA_BETA} Score (precision-weighted) : {fb:.4f}")
print(f"\n  → F-beta with beta=0.5 weights precision twice as much as recall.")
print(f"    Flagging a real customer's transaction as fraud has a business cost.")
print(f"    We penalise false positives more than missed fraud.")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Model Evaluation', fontsize=14, fontweight='bold')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
    xticklabels=['Pred: Legit', 'Pred: Fraud'],
    yticklabels=['True: Legit', 'True: Fraud']
)
axes[0].set_title('Confusion Matrix', fontweight='bold')

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
axes[1].plot(recall, precision, color='#e74c3c', linewidth=2.5,
             label=f'PR Curve  (AP = {ap:.3f})')
axes[1].fill_between(recall, precision, alpha=0.08, color='#e74c3c')
axes[1].set_xlabel('Recall  (What fraction of frauds did we catch?)')
axes[1].set_ylabel('Precision  (Of those flagged, how many were real fraud?)')
axes[1].set_title('Precision-Recall Curve\n(Better than ROC-AUC for imbalanced data)',
                  fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('02_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()
print("  → Saved: 02_evaluation.png")


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT 8 — SHAP GLOBAL FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  UNIT 8 — SHAP: GLOBAL FEATURE IMPORTANCE")
print("=" * 65)

explainer = shap.TreeExplainer(model)

# Use a sample to keep computation fast
sample_size = min(500, len(X_test))
X_sample = X_test.iloc[:sample_size]

print(f"  Computing SHAP values for {sample_size} test transactions...")
shap_values = explainer.shap_values(X_sample)

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False)
plt.title('Global Feature Importance — SHAP Mean |value|', fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('03_shap_global.png', dpi=150, bbox_inches='tight')
plt.show()
print("  → Saved: 03_shap_global.png")

# Dot/beeswarm plot (shows direction of effect too)
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_sample, show=False)
plt.title('SHAP Summary — Feature Effect Direction', fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('04_shap_summary.png', dpi=150, bbox_inches='tight')
plt.show()
print("  → Saved: 04_shap_summary.png")


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT 9 — RISK-TIER SCORING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  UNIT 9 — RISK-TIER SCORING ENGINE")
print("=" * 65)

TIERS = [
    (TIER_LOW_MAX,  "🟢 LOW RISK",    "Auto-approve"),
    (TIER_MED_MAX,  "🟡 MEDIUM RISK", "Flag for human review"),
    (101,           "🔴 HIGH RISK",   "Auto-block"),
]

def get_tier(score: int):
    for ceiling, label, action in TIERS:
        if score <= ceiling:
            return label, action

def explain_transaction(row, txn_id=None):
    """
    Score a single transaction and explain which features drove the decision.

    Parameters
    ----------
    row     : pd.Series or single-row pd.DataFrame
    txn_id  : optional label for display

    Returns
    -------
    dict with keys: score, tier_label, tier_action, fraud_prob, shap_top5
    """
    if isinstance(row, pd.Series):
        X_row = row[FEATURES].to_frame().T
    else:
        X_row = row[FEATURES]

    prob  = float(model.predict_proba(X_row)[0][1])
    score = int(round(prob * 100))
    tier_label, tier_action = get_tier(score)

    sv = explainer.shap_values(X_row)[0]
    shap_df = (
        pd.DataFrame({'feature': FEATURES, 'shap': sv})
        .assign(abs_shap=lambda d: d['shap'].abs())
        .sort_values('abs_shap', ascending=False)
        .head(5)
        .reset_index(drop=True)
    )

    sep = "─" * 56
    label = f"Transaction #{txn_id}" if txn_id is not None else "Transaction"

    print(f"\n  {sep}")
    print(f"  {label}")
    print(f"  {sep}")
    print(f"  Risk Score  : {score} / 100")
    print(f"  Risk Tier   : {tier_label}")
    print(f"  Action      : {tier_action}")
    print(f"  Fraud Prob  : {prob:.4f}")
    print(f"  {sep}")
    print("  Top contributing features:")
    print()
    for _, r in shap_df.iterrows():
        direction = "▲ raises fraud risk" if r['shap'] > 0 else "▼ lowers fraud risk"
        bar = "█" * min(int(abs(r['shap']) * 20), 20)
        print(f"  {r['feature']:<18} {direction:<26} {r['shap']:+.4f}  {bar}")
    print(f"  {sep}")

    return {
        'score': score,
        'tier_label': tier_label,
        'tier_action': tier_action,
        'fraud_prob': round(prob, 4),
        'shap_top5': shap_df
    }


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT 10 — DEMO: EXPLAIN SAMPLE TRANSACTIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  UNIT 10 — DEMO: PER-TRANSACTION EXPLANATIONS")
print("=" * 65)

fraud_rows  = X_test[y_test == 1].iloc[:3]
legit_rows  = X_test[y_test == 0].iloc[:2]

print("\n  ━━━ FRAUDULENT TRANSACTIONS ━━━")
for idx in fraud_rows.index:
    explain_transaction(X_test.loc[idx], txn_id=idx)

print("\n  ━━━ LEGITIMATE TRANSACTIONS ━━━")
for idx in legit_rows.index:
    explain_transaction(X_test.loc[idx], txn_id=idx)


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT 11 — SHAP WATERFALL PLOT (Interview-ready visual)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  UNIT 11 — SHAP WATERFALL PLOT")
print("=" * 65)

fraud_row_df = X_test[y_test == 1].iloc[[0]]
explanation  = explainer(fraud_row_df)

plt.figure(figsize=(11, 6))
shap.plots.waterfall(explanation[0], max_display=12, show=False)
plt.title('Why This Transaction Was Flagged — SHAP Waterfall',
          fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('05_shap_waterfall.png', dpi=150, bbox_inches='tight')
plt.show()
print("  → Saved: 05_shap_waterfall.png")

# ═══════════════════════════════════════════════════════════════════════════════
# UNIT 12 — TIER DISTRIBUTION ON TEST SET
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  UNIT 12 — RISK TIER DISTRIBUTION")
print("=" * 65)

all_proba  = model.predict_proba(X_test)[:, 1]
all_scores = (all_proba * 100).astype(int)

tier_labels = []
for s in all_scores:
    label, _ = get_tier(int(s))
    tier_labels.append(label)

tier_series = pd.Series(tier_labels)
tier_counts = tier_series.value_counts().reindex([
    "🟢 LOW RISK", "🟡 MEDIUM RISK", "🔴 HIGH RISK"
]).fillna(0).astype(int)

print("\n  Risk tier distribution across test set:")
for tier, count in tier_counts.items():
    pct = count / len(X_test) * 100
    print(f"  {tier:<22} {count:>6,} transactions  ({pct:.2f}%)")

colors = ['#2ecc71', '#f39c12', '#e74c3c']
plt.figure(figsize=(8, 5))
bars = plt.bar(
    ['Low Risk\n(Auto-approve)', 'Medium Risk\n(Human review)', 'High Risk\n(Auto-block)'],
    tier_counts.values, color=colors, edgecolor='black', linewidth=0.7
)
for bar, val in zip(bars, tier_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, val + 20,
             f'{val:,}', ha='center', fontweight='bold')
plt.title('Transaction Risk Tier Distribution — Test Set', fontweight='bold')
plt.ylabel('Transaction Count')
plt.tight_layout()
plt.savefig('06_tier_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("  → Saved: 06_tier_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  ✅  COMPLETE")
print("=" * 65)
print("""
  Outputs generated:
    01_eda.png               — Class imbalance, amount & time distributions
    02_evaluation.png        — Confusion matrix + precision-recall curve
    03_shap_global.png       — Global SHAP feature importance (bar)
    04_shap_summary.png      — SHAP beeswarm (shows direction of effect)
    05_shap_waterfall.png    — Single-transaction SHAP explanation
    06_tier_distribution.png — Risk tier breakdown across test set

  Key design decisions you can defend:
    1. Split BEFORE SMOTE          — prevents data leakage into test set
    2. sampling_strategy=0.10      — realistic ratio, not artificial 50/50
    3. eval_metric='aucpr'         — precision-recall AUC, correct for imbalanced data
    4. F-beta (beta=0.5)           — penalises false positives more than missed fraud
    5. Three-tier scoring          — business-aligned: not all fraud is auto-block
    6. SHAP per transaction        — explainability is a fintech compliance requirement
""")
