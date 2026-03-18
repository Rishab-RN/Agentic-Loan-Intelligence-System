"""
ALIS Fairness Audit: Fairlearn Core Engine
======================================================
1. Loads demographic synthetic data
2. Trains base XGBoost model (BLIND to demographics)
3. Computes Fairlearn MetricFrame (Demographic Parity, Equalized Odds)
4. If disparity > 5%, applies ExponentiatedGradient mitigation
5. Saves results and plots for the report generator

Usage:
    python audit.py
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
)
from fairlearn.reductions import DemographicParity, EqualizedOdds, ExponentiatedGradient
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from data_generator import ARTIFACTS_DIR

MODEL_DIR = ARTIFACTS_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = ARTIFACTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def run_audit():
    print("ALIS Fairness Audit: Running Fairlearn Analysis...")
    
    # ─── 1. Load Data ────────────────────────────────────────────────────────
    data_path = ARTIFACTS_DIR / "demographic_dataset.csv"
    if not data_path.exists():
        print("Demographic dataset not found. Run data_generator.py first.")
        from data_generator import generate_demographic_dataset
        df = generate_demographic_dataset()
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    # ─── 2. Prepare Training data ─────────────────────────────────────────────
    # Target
    y = df["loan_approved"]
    
    # Sensitive attributes (NOT for training!)
    A_gender = df["gender"]
    A_tier = df["geography_tier"]
    
    # Features
    X = df.drop(columns=[
        "loan_approved", "gender", "geography_tier", "income_band", "occupation"
    ])

    X_train, X_test, y_train, y_test, A_gender_train, A_gender_test, A_tier_train, A_tier_test = train_test_split(
        X, y, A_gender, A_tier, test_size=0.3, random_state=42, stratify=y
    )

    # ─── 3. Train Base (Unmitigated) Model ────────────────────────────────────
    print("\nTraining Base XGBoost model (blind to demographics)...")
    base_model = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42,
        eval_metric="logloss"
    )
    base_model.fit(X_train, y_train)
    y_pred_base = base_model.predict(X_test)
    
    # Base accuracy
    acc_base = accuracy_score(y_test, y_pred_base)
    print(f"Base Model Accuracy: {acc_base:.3f}")

    # ─── 4. Fairlearn MetricFrame Analysis (Gender) ───────────────────────────
    print("\nCreating Fairlearn MetricFrame for Gender...")
    metrics = {
        "accuracy": accuracy_score,
        "selection_rate": selection_rate,  # Approval Rate
        "fpr": false_positive_rate,
        "fnr": false_negative_rate
    }
    
    mf_base = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred_base,
        sensitive_features=A_gender_test
    )
    
    print("\nBase Model Metrics by Gender:")
    print(mf_base.by_group)
    
    dp_diff = demographic_parity_difference(y_test, y_pred_base, sensitive_features=A_gender_test)
    eo_diff = equalized_odds_difference(y_test, y_pred_base, sensitive_features=A_gender_test)
    
    print(f"\nDemographic Parity Difference: {dp_diff:.3f}")
    print(f"Equalized Odds Difference: {eo_diff:.3f}")
    
    # ─── 5. Visualize Base Disparities ────────────────────────────────────────
    plt.style.use('dark_background')
    
    mf_base.by_group[["selection_rate", "fpr", "fnr"]].plot(
        kind='bar', figsize=(10, 6), colormap='Set2'
    )
    plt.title("Loan Approval Metrics by Gender (Base Unmitigated Model)")
    plt.ylabel("Rate")
    plt.xticks(rotation=0)
    plt.legend(["Approval Rate", "False Positive Rate (FPR)", "False Negative Rate (FNR)"])
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "metrics_by_gender_base.png", dpi=300)
    plt.close()

    # ─── 6. Mitigation using ExponentiatedGradient ────────────────────────────
    # If DP difference > 5% (0.05), apply mitigation
    mitigated_model = None
    y_pred_mitigated = None
    if dp_diff > 0.05:
        print("\n⚠️ Demographic disparity > 5% detected. Applying ExponentiatedGradient mitigation...")
        
        # We enforce EqualizedOdds (fairness constraint)
        constraint = EqualizedOdds()
        mitigator = ExponentiatedGradient(
            # We use the same base_model class
            estimator=XGBClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, eval_metric="logloss"
            ),
            constraints=constraint,
            eps=0.015,
            max_iter=30
        )
        
        # Train mitigator using sensitive features during training
        mitigator.fit(X_train, y_train, sensitive_features=A_gender_train)
        y_pred_mitigated = mitigator.predict(X_test)
        
        acc_mit = accuracy_score(y_test, y_pred_mitigated)
        mf_mit = MetricFrame(
            metrics=metrics,
            y_true=y_test,
            y_pred=y_pred_mitigated,
            sensitive_features=A_gender_test
        )
        
        dp_diff_mit = demographic_parity_difference(y_test, y_pred_mitigated, sensitive_features=A_gender_test)
        eo_diff_mit = equalized_odds_difference(y_test, y_pred_mitigated, sensitive_features=A_gender_test)
        
        print("\nMitigated Model Metrics by Gender:")
        print(mf_mit.by_group)
        print(f"\nMitigated Demographic Parity Difference: {dp_diff_mit:.3f}")
        print(f"Mitigated Equalized Odds Difference: {eo_diff_mit:.3f}")
        print(f"Accuracy-Fairness Tradeoff: Accuracy dropped from {acc_base:.3f} to {acc_mit:.3f}")

        # Visualize mitigated disparities
        mf_mit.by_group[["selection_rate", "fpr", "fnr"]].plot(
            kind='bar', figsize=(10, 6), colormap='Set2'
        )
        plt.title("Loan Approval Metrics by Gender (Mitigated Model)")
        plt.ylabel("Rate")
        plt.xticks(rotation=0)
        plt.legend(["Approval Rate", "False Positive Rate (FPR)", "False Negative Rate (FNR)"])
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "metrics_by_gender_mitigated.png", dpi=300)
        plt.close()
    
        # Save metrics dictionary for report generator
        metrics_dict = {
            "unmitigated": {
                "accuracy": float(acc_base),
                "dp_diff": float(dp_diff),
                "eo_diff": float(eo_diff),
                "by_group": mf_base.by_group.to_dict()
            },
            "mitigated": {
                "accuracy": float(acc_mit),
                "dp_diff": float(dp_diff_mit),
                "eo_diff": float(eo_diff_mit),
                "by_group": mf_mit.by_group.to_dict()
            }
        }
    else:
        print("\n✅ Base model meets RBI fairness thresholds. No mitigation required.")
        metrics_dict = {
            "unmitigated": {
                "accuracy": float(acc_base),
                "dp_diff": float(dp_diff),
                "eo_diff": float(eo_diff),
                "by_group": mf_base.by_group.to_dict()
            },
            "mitigated": None
        }

    # Save metrics JSON
    import json
    with open(ARTIFACTS_DIR / "audit_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)
        
    print("\nAudit complete. Metrics and plots saved to artifacts/")


if __name__ == "__main__":
    run_audit()
