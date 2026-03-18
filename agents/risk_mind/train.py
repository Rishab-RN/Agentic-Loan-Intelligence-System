"""
ALIS — RiskMind: Model Training Pipeline
==========================================
Trains an XGBoost credit-scoring classifier on the synthetic gig-worker
dataset, performs cross-validation, evaluates on a held-out test set, and
saves the trained model + scaler for production serving.

Outputs (saved to artifacts/):
    - riskmind_model.joblib          XGBoost classifier
    - riskmind_scaler.joblib         StandardScaler fitted on training data
    - training_report.txt            Full classification report + metrics

Usage:
    python train.py                  # uses default dataset
    python train.py --data path.csv  # custom dataset
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data_generator import FEATURE_COLUMNS, TARGET_COLUMN, generate_gig_worker_dataset

# ─── Configuration ───────────────────────────────────────────────────────────

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,          # L1 regularization
    "reg_lambda": 1.0,         # L2 regularization
    "scale_pos_weight": 1.0,   # adjusted dynamically below
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}

TEST_SIZE = 0.20
CV_FOLDS = 5
RANDOM_STATE = 42


# ─── Training Pipeline ──────────────────────────────────────────────────────

def load_or_generate_data(data_path: str | None = None) -> pd.DataFrame:
    """Load CSV if provided, otherwise generate fresh synthetic data."""
    if data_path and Path(data_path).exists():
        print(f"  Loading data from {data_path}")
        return pd.read_csv(data_path)

    print("  No dataset found — generating 5000 synthetic profiles...")
    df = generate_gig_worker_dataset(n=5000, seed=RANDOM_STATE)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    csv_path = ARTIFACTS_DIR / "synthetic_gig_workers.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved to {csv_path}")
    return df


def train_model(df: pd.DataFrame) -> dict:
    """
    Full training pipeline:
      1. Train/test split (stratified)
      2. Feature scaling
      3. Handle class imbalance via scale_pos_weight
      4. XGBoost training
      5. Platt-scaling calibration for reliable probabilities
      6. 5-fold cross-validation
      7. Test-set evaluation
      8. Save artifacts

    Returns a dict of metrics.
    """
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # ── 1. Stratified split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"\n  Train set: {len(X_train):,} samples  |  Test set: {len(X_test):,} samples")
    print(f"  Train approval rate: {y_train.mean():.1%}  |  Test approval rate: {y_test.mean():.1%}")

    # ── 2. Scaling ───────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURE_COLUMNS, index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURE_COLUMNS, index=X_test.index,
    )

    # ── 3. Class imbalance adjustment ────────────────────────────────────────
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"  Class balance — rejected: {neg_count}, approved: {pos_count}")
    print(f"  scale_pos_weight: {scale_pos_weight:.3f}")

    params = {**MODEL_PARAMS, "scale_pos_weight": scale_pos_weight}

    # ── 4. Train XGBoost ─────────────────────────────────────────────────────
    print("\n  Training XGBoost classifier...")
    base_model = XGBClassifier(**params)
    base_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False,
    )

    # ── 5. Probability calibration (Platt scaling) ───────────────────────────
    #   Raw XGBoost probabilities can be miscalibrated.  We wrap with
    #   CalibratedClassifierCV(method="sigmoid") for reliable credit scores.
    print("  Calibrating probabilities (Platt scaling)...")
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model, method="sigmoid", cv=3,
    )
    calibrated_model.fit(X_train_scaled, y_train)

    # ── 6. Cross-validation on training data ─────────────────────────────────
    print(f"  Running {CV_FOLDS}-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        base_model, X_train_scaled, y_train, cv=cv, scoring="roc_auc",
    )
    print(f"  CV AUC-ROC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── 7. Test-set evaluation ───────────────────────────────────────────────
    y_pred = calibrated_model.predict(X_test_scaled)
    y_prob = calibrated_model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "auc_roc": float(roc_auc_score(y_test, y_prob)),
        "cv_auc_mean": float(cv_scores.mean()),
        "cv_auc_std": float(cv_scores.std()),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }

    # Print report
    report = classification_report(y_test, y_pred, target_names=["Rejected", "Approved"])
    print(f"\n{'='*60}")
    print("  CLASSIFICATION REPORT (Test Set)")
    print(f"{'='*60}")
    print(report)
    print(f"  AUC-ROC:  {metrics['auc_roc']:.4f}")
    print(f"  CV AUC:   {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
    print(f"{'='*60}\n")

    # ── 8. Save artifacts ────────────────────────────────────────────────────
    # Save the calibrated model (production) and the base model (for SHAP)
    joblib.dump(calibrated_model, ARTIFACTS_DIR / "riskmind_model.joblib")
    joblib.dump(base_model, ARTIFACTS_DIR / "riskmind_base_model.joblib")
    joblib.dump(scaler, ARTIFACTS_DIR / "riskmind_scaler.joblib")

    # Save report
    report_path = ARTIFACTS_DIR / "training_report.txt"
    with open(report_path, "w") as f:
        f.write("ALIS RiskMind — Training Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset size:  {len(df):,}\n")
        f.write(f"Train size:    {len(X_train):,}\n")
        f.write(f"Test size:     {len(X_test):,}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write(f"AUC-ROC:       {metrics['auc_roc']:.4f}\n")
        f.write(f"CV AUC:        {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}\n")

    # Save metrics as JSON for programmatic access
    with open(ARTIFACTS_DIR / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Model saved:   {ARTIFACTS_DIR / 'riskmind_model.joblib'}")
    print(f"  Scaler saved:  {ARTIFACTS_DIR / 'riskmind_scaler.joblib'}")
    print(f"  Report saved:  {report_path}")

    return metrics


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ALIS RiskMind — Train XGBoost credit scoring model"
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to training CSV (default: auto-generate synthetic data)",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  ALIS RiskMind — Model Training Pipeline")
    print(f"{'='*60}")

    df = load_or_generate_data(args.data)
    metrics = train_model(df)

    print("\n  ✓ Training complete.\n")


if __name__ == "__main__":
    main()
