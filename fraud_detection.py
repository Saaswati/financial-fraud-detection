# =============================================================================
# FINANCIAL FRAUD DETECTION SYSTEM
# Risk-Tiered Scoring Engine with SHAP Per-Transaction Explanations
#
# Dataset : Kaggle Credit Card Fraud Detection
#           kaggle.com/datasets/mlg-ulb/creditcardfraud
# Author  : Saaswati
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    fbeta_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
DATA_PATH        = os.environ.get('CREDITCARD_CSV', 'creditcard.csv')
RANDOM_STATE     = 42
TEST_SIZE        = 0.20   # 20% of all data  → final, untouched test set
VAL_SIZE         = 0.25   # 25% of the rest  → validation set (= 20% of all data)
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

df = pd.read_csv(DATA_PATH)

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

# 2b. Amount distribution (dataset is European cardholders → amounts in euros)
axes[1].hist(df[df['Class']==0]['Amount'], bins=60, alpha=0.6,
             color='#2ecc71', label='Legitimate', density=True)
axes[1].hist(df[df['Class']==1]['Amount'], bins=60, alpha=0.6,
             color='#e74c3c', label='Fraud', density=True)
axes[1].set_title('Transaction Amount Distribution', fontweight='bold')
axes[1].set_xlabel('Amount (€)')
axes[1].set_xlim(0, 2500)
axes[1].legend()

# 2c. Time distribution (hours since the first transaction — NOT hour of day)
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
# UNIT 3 — TRAIN / VALIDATION / TEST SPLIT  ← BEFORE ANY FITTING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  UNIT 3 — TRAIN / VALIDATION / TEST SPLIT")
print("=" * 65)

# WHY SPLIT FIRST:
# Anything that "learns" from data — the scaler's mean/std, SMOTE's synthetic
# points, the model, the calibrator, the decision threshold — must learn from
# training (or validation) data only. If the test set influences any of them,
# test scores become optimistic. This is data leakage.
#
# WHY THREE SETS:
#   train      → fit scaler, SMOTE, XGBoost
#   validation → early stopping, probability calibration, threshold choice
#   test       → touched ONCE, at the end, for the final honest numbers

RAW_FEATURES = [c for c in df.columns if c != 'Class']
X = df[RAW_FEATURES]
y = df['Class']

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y          # preserves the fraud ratio in both splits
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=VAL_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_trainval
)

print(f"  Train      : {X_train.shape[0]:>7,} rows | Fraud: {y_train.sum():>4}")
print(f"  Validation : {X_val.shape[0]:>7,} rows | Fraud: {y_val.sum():>4}")
print(f"  Test       : {X_test.shape[0]:>7,} rows | Fraud: {y_test.sum():>4}")


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT 4 — PREPROCESSING (scaler fitted on TRAINING data only)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  UNIT 4 — PREPROCESSING")
print("=" * 65)

# V1–V28 are already PCA-transformed (centred). Amount and Time are raw.
# Tree models don't need scaling (splits are threshold-based), but SMOTE does:
# it interpolates between nearest neighbours, and neighbours are found by
# distance. Unscaled, Time (0–172,792 seconds) would dominate every distance.
#
# One scaler for both columns: StandardScaler stores a separate mean/std per
# column, and the same fitted object can later be saved and reused in production.

SCALE_COLS = ['Amount', 'Time']
scaler = StandardScaler().fit(X_train[SCALE_COLS])

def preprocess(X_part: pd.DataFrame) -> pd.DataFrame:
    """Apply the training-fitted scaler. Never refits."""
    out = X_part.drop(columns=SCALE_COLS).copy()
    scaled = scaler.transform(X_part[SCALE_COLS])
    out['Amount_scaled'] = scaled[:, 0]
    out['Time_scaled']   = scaled[:, 1]
    return out

X_train = preprocess(X_train)
X_val   = preprocess(X_val)
X_test  = preprocess(X_test)
FEATURES = list(X_train.columns)

print(f"  Feature count : {len(FEATURES)}")
print(f"  → Scaler fitted on training rows only, then applied to val/test.")


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT 5 — SMOTE (training data only)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  UNIT 5 — SMOTE (Synthetic Minority Oversampling)")
print("=" * 65)

# WHY NOT 50/50 SMOTE:
# Upsampling fraud to 50% of all transactions is unrealistic — the model
# learns a world that doesn't exist. Setting sampling_strategy=0.1 means
# "after SMOTE, fraud count = 10% of non-fraud count." Still much better
# than 0.17%, but not an artificial world.

smote = SMOTE(sampling_strategy=SMOTE_RATIO, random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"  Before SMOTE → Fraud: {y_train.sum():,}  |  Legit: {(y_train==0).sum():,}")
print(f"  After SMOTE  → Fraud: {y_train_res.sum():,}  |  Legit: {(y_train_res==0).sum():,}")
print(f"  New fraud rate in training set: {y_train_res.mean()*100:.2f}%")
print(f"\n  ⚠  Validation and test sets = real transactions only.")


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT 6 — XGBOOST TRAINING (early stopping on the validation set)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  UNIT 6 — XGBOOST TRAINING")
print("=" * 65)

# WHY XGBOOST OVER RANDOM FOREST / LOGISTIC REGRESSION:
# XGBoost builds trees sequentially, each correcting the errors of the last.
# It captures non-linear feature interactions, is robust to outliers, and is
# state-of-the-art on tabular data. SHAP's TreeExplainer computes exact
# SHAP values for it quickly.
#
# WHY NO scale_pos_weight:
# SMOTE already lifts fraud to ~9% of training rows. Adding scale_pos_weight≈10
# on top multiplies each fraud's weight by 10 → effectively a 50/50 world again,
# which is exactly what sampling_strategy=0.10 was chosen to avoid. It also makes
# the predicted probabilities overconfident. One imbalance fix, not two.
#
# WHY A VALIDATION SET FOR eval_set:
# Early stopping picks the number of trees by watching eval_set. If that were
# the test set, the test set would be tuning the model → leakage.

model = xgb.XGBClassifier(
    n_estimators          = 1000,     # upper bound; early stopping picks the real number
    max_depth             = 6,
    learning_rate         = 0.05,
    subsample             = 0.8,      # row subsampling per tree → prevents overfitting
    colsample_bytree      = 0.8,      # feature subsampling per tree → prevents overfitting
    eval_metric           = 'aucpr',  # area under precision-recall curve — right for imbalanced data
    early_stopping_rounds = 50,       # stop if val PR-AUC hasn't improved for 50 trees
    random_state          = RANDOM_STATE,
    n_jobs                = -1
)

model.fit(
    X_train_res, y_train_res,
    eval_set=[(X_val, y_val)],
    verbose=100
)

print(f"\n  ✓ Training complete. Best iteration: {model.best_iteration} "
      f"(val PR-AUC = {model.best_score:.4f})")


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT 7 — PROBABILITY CALIBRATION + THRESHOLD SELECTION (validation set)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  UNIT 7 — CALIBRATION & THRESHOLD")
print("=" * 65)

# WHY CALIBRATE:
# The model was trained on ~9% fraud (SMOTE), but reality is ~0.17%. Its raw
# probabilities are therefore inflated. The risk score is "probability × 100",
# so the probability must mean what it says: a score of 30 should be right
# about 30% of the time. Isotonic regression learns a monotonic mapping
# raw_prob → real fraud frequency, fitted on the (real-distribution) validation set.
# Monotonic = transaction ranking is unchanged, so PR-AUC and SHAP directions hold.

val_raw = model.predict_proba(X_val)[:, 1]
calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
calibrator.fit(val_raw, y_val)

def predict_fraud_proba(X_part: pd.DataFrame) -> np.ndarray:
    """Calibrated fraud probability — use this everywhere, not model.predict_proba."""
    return calibrator.predict(model.predict_proba(X_part)[:, 1])

def to_score(prob) -> int:
    """Single source of truth for probability → 0–100 risk score."""
    return int(round(float(prob) * 100))

# WHY TUNE THE THRESHOLD:
# model.predict() silently uses 0.5. We report F-beta, so we should pick the
# threshold that maximises F-beta — on validation data, never on test data.
val_cal = predict_fraud_proba(X_val)
prec_v, rec_v, thr_v = precision_recall_curve(y_val, val_cal)
b2 = FBETA_BETA ** 2
fbeta_v = (1 + b2) * prec_v * rec_v / np.clip(b2 * prec_v + rec_v, 1e-12, None)
best = int(np.argmax(fbeta_v[:-1]))          # last PR point has no threshold
THRESHOLD = float(thr_v[best])

print(f"  Calibrator fitted on {len(y_val):,} validation transactions.")
print(f"  Mean raw prob (val)        : {val_raw.mean():.4f}")
print(f"  Mean calibrated prob (val) : {val_cal.mean():.4f}")
print(f"  Actual fraud rate (val)    : {y_val.mean():.4f}")
print(f"\n  Best F-{FBETA_BETA} threshold on validation : {THRESHOLD:.3f} "
      f"(val F-{FBETA_BETA} = {fbeta_v[best]:.4f})")


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT 8 — EVALUATION (test set — first and only time it's used)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  UNIT 8 — EVALUATION ON TEST SET")
print("=" * 65)

y_proba = predict_fraud_proba(X_test)
y_pred  = (y_proba >= THRESHOLD).astype(int)

ap    = average_precision_score(y_test, y_proba)
roc   = roc_auc_score(y_test, y_proba)
fb    = fbeta_score(y_test, y_pred, beta=FBETA_BETA)

print(f"\n  Classification Report (threshold = {THRESHOLD:.3f}):")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud'], digits=4))
print(f"  Average Precision (PR-AUC)       : {ap:.4f}")
print(f"  ROC-AUC (for comparison only)    : {roc:.4f}")
print(f"  F-{FBETA_BETA} Score (precision-weighted) : {fb:.4f}")
print(f"\n  → F-beta with beta=0.5 treats precision as more important than recall.")
print(f"    Flagging a real customer's transaction as fraud has a business cost.")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Model Evaluation — Test Set', fontsize=14, fontweight='bold')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt=',d', cmap='Blues', ax=axes[0],
    xticklabels=['Pred: Legit', 'Pred: Fraud'],
    yticklabels=['True: Legit', 'True: Fraud']
)
axes[0].set_title(f'Confusion Matrix (threshold = {THRESHOLD:.2f})', fontweight='bold')

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
axes[1].plot(recall, precision, color='#e74c3c', linewidth=2.5,
             label=f'PR Curve  (AP = {ap:.3f})')
axes[1].fill_between(recall, precision, alpha=0.08, color='#e74c3c')
axes[1].axhline(y_test.mean(), color='grey', linestyle='--', linewidth=1,
                label=f'Random baseline ({y_test.mean():.4f})')
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
# UNIT 9 — SHAP GLOBAL FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  UNIT 9 — SHAP: GLOBAL FEATURE IMPORTANCE")
print("=" * 65)

# SHAP explains the XGBoost model's raw output (log-odds). Calibration is a
# monotonic mapping applied afterwards, so which features push risk up or down
# — and by how much relative to each other — is unchanged.
explainer = shap.TreeExplainer(model)

# Random sample (not just the first rows) to keep computation fast
sample_size = min(2000, len(X_test))
X_sample = X_test.sample(n=sample_size, random_state=RANDOM_STATE)

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
# UNIT 10 — RISK-TIER SCORING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  UNIT 10 — RISK-TIER SCORING ENGINE")
print("=" * 65)

TIERS = [
    (TIER_LOW_MAX,  "🟢 LOW RISK",    "Auto-approve"),
    (TIER_MED_MAX,  "🟡 MEDIUM RISK", "Flag for human review"),
    (100,           "🔴 HIGH RISK",   "Auto-block"),
]
TIER_ORDER = [label for _, label, _ in TIERS]

def get_tier(score: int):
    for ceiling, label, action in TIERS:
        if score <= ceiling:
            return label, action

def explain_transaction(row, txn_id=None):
    """
    Score a single transaction and explain which features drove the decision.

    Parameters
    ----------
    row     : pd.Series or single-row pd.DataFrame (already preprocessed)
    txn_id  : optional label for display

    Returns
    -------
    dict with keys: score, tier_label, tier_action, fraud_prob, shap_top5
    """
    if isinstance(row, pd.Series):
        X_row = row[FEATURES].to_frame().T.astype(float)
    else:
        X_row = row[FEATURES]

    prob  = float(predict_fraud_proba(X_row)[0])
    score = to_score(prob)
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
    print(f"  Fraud Prob  : {prob:.4f}  (calibrated)")
    print(f"  {sep}")
    print("  Top contributing features (SHAP, log-odds):")
    print()
    for _, r in shap_df.iterrows():
        direction = "▲ raises fraud risk" if r['shap'] > 0 else "▼ lowers fraud risk"
        bar = "█" * min(int(abs(r['shap']) * 5), 20)
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
# UNIT 11 — DEMO: EXPLAIN SAMPLE TRANSACTIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  UNIT 11 — DEMO: PER-TRANSACTION EXPLANATIONS")
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
# UNIT 12 — SHAP WATERFALL PLOT (Interview-ready visual)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  UNIT 12 — SHAP WATERFALL PLOT")
print("=" * 65)

fraud_row_df = X_test[y_test == 1].iloc[[0]]
explanation  = explainer(fraud_row_df)

plt.figure(figsize=(11, 6))
shap.plots.waterfall(explanation[0], max_display=12, show=False)
plt.title('Why This Transaction Was Flagged — SHAP Waterfall (log-odds)',
          fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('05_shap_waterfall.png', dpi=150, bbox_inches='tight')
plt.show()
print("  → Saved: 05_shap_waterfall.png")


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT 13 — TIER DISTRIBUTION ON TEST SET
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  UNIT 13 — RISK TIER DISTRIBUTION")
print("=" * 65)

all_scores  = [to_score(p) for p in y_proba]      # same rounding as explain_transaction
tier_labels = pd.Series([get_tier(s)[0] for s in all_scores], index=y_test.index)

tier_by_class = (
    pd.crosstab(tier_labels, y_test.map({0: 'Legitimate', 1: 'Fraud'}))
    .reindex(index=TIER_ORDER, columns=['Legitimate', 'Fraud'])
    .fillna(0).astype(int)
    .rename_axis(index='Risk tier', columns='Actual')
)
tier_counts = tier_by_class.sum(axis=1)

print("\n  Risk tier distribution across test set:")
for tier, count in tier_counts.items():
    pct = count / len(X_test) * 100
    print(f"  {tier:<22} {count:>6,} transactions  ({pct:.2f}%)")
print("\n  Tier × actual class (where does the real fraud land?):")
print(tier_by_class.to_string())

colors = ['#2ecc71', '#f39c12', '#e74c3c']
plt.figure(figsize=(8, 5))
bars = plt.bar(
    ['Low Risk\n(Auto-approve)', 'Medium Risk\n(Human review)', 'High Risk\n(Auto-block)'],
    tier_counts.values, color=colors, edgecolor='black', linewidth=0.7
)
plt.yscale('log')   # low-risk tier is ~1000× larger than the others
for bar, tier in zip(bars, TIER_ORDER):
    total = tier_counts[tier]
    fraud = tier_by_class.loc[tier, 'Fraud']
    plt.text(bar.get_x() + bar.get_width()/2, max(total, 1) * 1.15,
             f'{total:,}\n({fraud} fraud)', ha='center', fontweight='bold')
plt.ylim(top=tier_counts.max() * 8)
plt.title('Transaction Risk Tier Distribution — Test Set', fontweight='bold')
plt.ylabel('Transaction Count (log scale)')
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
    1. Split FIRST (train/val/test) — scaler, SMOTE, model all fit on train only
    2. sampling_strategy=0.10      — realistic ratio, not artificial 50/50
    3. No scale_pos_weight         — one imbalance fix, not two stacked
    4. Early stopping on val set   — test set never influences training
    5. Isotonic calibration        — risk score reflects real fraud probability
    6. F-beta-tuned threshold      — chosen on validation, not the default 0.5
    7. Three-tier scoring          — business-aligned: not all fraud is auto-block
    8. SHAP per transaction        — explainability is a fintech compliance requirement
""")
