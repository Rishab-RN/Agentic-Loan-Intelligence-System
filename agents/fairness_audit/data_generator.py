"""
ALIS Fairness Audit: Expanded Synthetic Data Generator
======================================================
Generates an expanded version of the RiskMind dataset that includes
sensitive demographic attributes for fairness auditing.

Demographics generated:
  - gender: male, female
  - geography_tier: tier1, tier2, tier3
  - income_band: low (<10k), medium (10-30k), high (>30k)
  - occupation: gig_delivery, auto_driver, tuition_teacher, kirana_owner, domestic_worker

IMPORTANT:
These demographic features are strictly for auditing and are NOT
passed to the XGBoost model during training/inference.

Usage:
    python data_generator.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse RiskMind's feature generation logic where possible
import sys
sys.path.append(str(Path(__file__).parent.parent))
from risk_mind.data_generator import generate_gig_worker_dataset

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_demographic_dataset(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic data with systemic disparities linked to demographics.
    This creates realistic biases that the Fairlearn audit can detect and mitigate.
    """
    np.random.seed(random_state)
    
    # Generate the base behavioral features and loan_approved target
    df = generate_gig_worker_dataset(n_samples=n_samples)
    
    # ─── 1. Generate Demographics with Realistic Indian Distributions ──────────
    
    # Gender: Historically skewed in gig economy (e.g., delivery, auto)
    # We'll assign gender based loosely on the generated behavioral "persona"
    # To do this without rewriting RiskMind, we'll infer persona-like traits
    # from the generated features, then assign demographics.
    
    genders = []
    tiers = []
    incomes = []
    occupations = []
    
    for _, row in df.iterrows():
        # Infer occupation archetype roughly based on features
        inc_est = row["income_estimate_monthly"]
        upi_freq = row["upi_txn_frequency_30d"]
        night_txn = row["evening_txn_ratio"]
        
        # Determine occupation and gender
        if upi_freq > 150 and night_txn > 0.4:
            occ = "gig_delivery"
            gender = np.random.choice(["male", "female"], p=[0.95, 0.05])
        elif inc_est > 25000 and upi_freq > 100:
            occ = "kirana_owner"
            gender = np.random.choice(["male", "female"], p=[0.80, 0.20])
        elif inc_est < 12000 and row["utility_bill_payment_consistency"] > 0.8:
            occ = "domestic_worker"
            gender = np.random.choice(["male", "female"], p=[0.05, 0.95])
        elif night_txn < 0.2 and row["savings_behavior_score"] > 60:
            occ = "tuition_teacher"
            gender = np.random.choice(["male", "female"], p=[0.30, 0.70])
        else:
            occ = "auto_driver"
            gender = np.random.choice(["male", "female"], p=[0.99, 0.01])
            
        occupations.append(occ)
        genders.append(gender)
        
        # Geography Tier
        # Tier 3 usually has lower UPI diversity and lower income
        if row["upi_merchant_diversity_score"] < 0.3 and inc_est < 15000:
            tier = "tier3_city"
        elif row["upi_merchant_diversity_score"] > 0.6:
            tier = "tier1_city"
        else:
            tier = "tier2_city"
        tiers.append(tier)
        
        # Income Band
        if inc_est < 10000:
            band = "low"
        elif inc_est <= 30000:
            band = "medium"
        else:
            band = "high"
        incomes.append(band)

    df["gender"] = genders
    df["geography_tier"] = tiers
    df["income_band"] = incomes
    df["occupation"] = occupations
    
    # ─── 2. Inject Systemic Bias ──────────────────────────────────────────────
    # To make the fairness audit meaningful, we ensure the underlying target
    # label (loan_approved) correlates with demographics because the behavioral
    # features were themselves correlated with demographics.
    # Women ("domestic_worker", "tuition_teacher") might have lower incomes
    # and lower UPI frequency, resulting in lower approval rates natively.
    
    # Overwrite a small percentage of female/tier3 approvals to be rejected
    # to explicitly simulate historical bias/underrepresentation in the training data
    mask_bias = (df["gender"] == "female") | (df["geography_tier"] == "tier3_city")
    indices_to_flip = df[mask_bias & (df["loan_approved"] == 1)].sample(frac=0.15, random_state=random_state).index
    df.loc[indices_to_flip, "loan_approved"] = 0

    return df


def main():
    print("Generating demographic-aware dataset for ALIS Fairness Audit...")
    df = generate_demographic_dataset(n_samples=1000)
    
    out_path = ARTIFACTS_DIR / "demographic_dataset.csv"
    df.to_csv(out_path, index=False)
    
    print(f"Dataset generated with {len(df)} records.")
    print("\nGender Breakdown:")
    print(df["gender"].value_counts(normalize=True))
    print("\nGeography Breakdown:")
    print(df["geography_tier"].value_counts(normalize=True))
    print("\nApproval Rate by Gender:")
    print(df.groupby("gender")["loan_approved"].mean())
    print("\nApproval Rate by Geography:")
    print(df.groupby("geography_tier")["loan_approved"].mean())


if __name__ == "__main__":
    main()
