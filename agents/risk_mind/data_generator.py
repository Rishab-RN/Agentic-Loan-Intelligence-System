"""
ALIS — RiskMind: Synthetic Data Generator
==========================================
Generates realistic synthetic profiles for India's gig-economy workers
(auto drivers, kirana owners, tuition teachers, domestic workers) with
alternative credit features derived from UPI, utility, and mobile data.

The target label is engineered with domain-informed rules — NOT random — so
the trained model learns genuinely predictive patterns.

Usage:
    python data_generator.py              # generates 5000 samples
    python data_generator.py --n 10000    # custom count
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Constants ───────────────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "upi_txn_frequency_30d",
    "upi_merchant_diversity_score",
    "utility_bill_payment_consistency",
    "mobile_recharge_regularity",
    "income_estimate_monthly",
    "income_volatility_cv",
    "bnpl_outstanding_ratio",
    "multi_loan_app_count",
    "evening_txn_ratio",
    "savings_behavior_score",
    "peer_transfer_reciprocity",
    "device_tenure_months",
]

TARGET_COLUMN = "loan_approved"

# Human-readable descriptions for SHAP explanations
FEATURE_DESCRIPTIONS = {
    "upi_txn_frequency_30d": "Daily UPI transaction frequency (last 30 days)",
    "upi_merchant_diversity_score": "Merchant spending diversity score",
    "utility_bill_payment_consistency": "Utility bill on-time payment rate (last 12 months)",
    "mobile_recharge_regularity": "Days between mobile recharges (lower = more regular)",
    "income_estimate_monthly": "Estimated monthly income from UPI credits (₹)",
    "income_volatility_cv": "Income stability (coefficient of variation — lower = more stable)",
    "bnpl_outstanding_ratio": "Buy-Now-Pay-Later balance vs. monthly income",
    "multi_loan_app_count": "Number of loan apps installed (financial distress signal)",
    "evening_txn_ratio": "Fraction of transactions after 8 PM (behavioral pattern)",
    "savings_behavior_score": "Savings transfer regularity score",
    "peer_transfer_reciprocity": "Balance of sent vs. received peer transfers (social trust)",
    "device_tenure_months": "Months using the current device (stability indicator)",
}


# ─── Persona Generators ─────────────────────────────────────────────────────

def _generate_auto_driver(rng: np.random.Generator) -> dict:
    """Auto-rickshaw driver in a Tier-2 city like Ballari."""
    income = rng.normal(18_000, 5_000)
    return {
        "upi_txn_frequency_30d": rng.uniform(3, 12),        # multiple rides/day
        "upi_merchant_diversity_score": rng.uniform(0.15, 0.40),  # fuel, food, limited shops
        "utility_bill_payment_consistency": rng.uniform(0.50, 0.92),
        "mobile_recharge_regularity": rng.uniform(7, 25),
        "income_estimate_monthly": max(income, 5_000),
        "income_volatility_cv": rng.uniform(0.25, 0.55),    # seasonal variation
        "bnpl_outstanding_ratio": rng.uniform(0.0, 0.25),
        "multi_loan_app_count": rng.choice([0, 0, 1, 1, 2]),
        "evening_txn_ratio": rng.uniform(0.25, 0.55),       # evening rides common
        "savings_behavior_score": rng.uniform(0.10, 0.50),
        "peer_transfer_reciprocity": rng.uniform(0.6, 1.3),
        "device_tenure_months": rng.uniform(6, 36),
    }


def _generate_kirana_owner(rng: np.random.Generator) -> dict:
    """Small kirana (grocery) shop owner."""
    income = rng.normal(25_000, 8_000)
    return {
        "upi_txn_frequency_30d": rng.uniform(8, 30),        # many small customer txns
        "upi_merchant_diversity_score": rng.uniform(0.05, 0.25),  # wholesale + few categories
        "utility_bill_payment_consistency": rng.uniform(0.65, 0.98),
        "mobile_recharge_regularity": rng.uniform(5, 15),
        "income_estimate_monthly": max(income, 8_000),
        "income_volatility_cv": rng.uniform(0.10, 0.35),    # relatively stable
        "bnpl_outstanding_ratio": rng.uniform(0.0, 0.15),
        "multi_loan_app_count": rng.choice([0, 0, 0, 1]),
        "evening_txn_ratio": rng.uniform(0.15, 0.40),
        "savings_behavior_score": rng.uniform(0.25, 0.70),
        "peer_transfer_reciprocity": rng.uniform(0.4, 0.9),
        "device_tenure_months": rng.uniform(12, 48),
    }


def _generate_tuition_teacher(rng: np.random.Generator) -> dict:
    """Freelance tuition teacher with variable monthly income."""
    income = rng.normal(15_000, 6_000)
    return {
        "upi_txn_frequency_30d": rng.uniform(1, 6),         # fee collections
        "upi_merchant_diversity_score": rng.uniform(0.20, 0.55),
        "utility_bill_payment_consistency": rng.uniform(0.70, 0.99),
        "mobile_recharge_regularity": rng.uniform(10, 28),
        "income_estimate_monthly": max(income, 4_000),
        "income_volatility_cv": rng.uniform(0.20, 0.50),    # summer months drop
        "bnpl_outstanding_ratio": rng.uniform(0.0, 0.20),
        "multi_loan_app_count": rng.choice([0, 0, 1]),
        "evening_txn_ratio": rng.uniform(0.10, 0.30),
        "savings_behavior_score": rng.uniform(0.30, 0.75),
        "peer_transfer_reciprocity": rng.uniform(0.7, 1.4),
        "device_tenure_months": rng.uniform(8, 42),
    }


def _generate_domestic_worker(rng: np.random.Generator) -> dict:
    """Domestic worker — lowest digital footprint, highest inclusion need."""
    income = rng.normal(10_000, 3_000)
    return {
        "upi_txn_frequency_30d": rng.uniform(0.5, 4),       # very few digital txns
        "upi_merchant_diversity_score": rng.uniform(0.05, 0.20),
        "utility_bill_payment_consistency": rng.uniform(0.40, 0.85),
        "mobile_recharge_regularity": rng.uniform(15, 35),
        "income_estimate_monthly": max(income, 3_000),
        "income_volatility_cv": rng.uniform(0.15, 0.40),
        "bnpl_outstanding_ratio": rng.uniform(0.0, 0.35),
        "multi_loan_app_count": rng.choice([0, 0, 0, 1, 2, 3]),
        "evening_txn_ratio": rng.uniform(0.05, 0.25),
        "savings_behavior_score": rng.uniform(0.05, 0.35),
        "peer_transfer_reciprocity": rng.uniform(0.8, 1.5),
        "device_tenure_months": rng.uniform(3, 24),
    }


def _generate_distressed_profile(rng: np.random.Generator) -> dict:
    """Deliberately high-risk profile: over-leveraged, distressed signals."""
    income = rng.normal(12_000, 4_000)
    return {
        "upi_txn_frequency_30d": rng.uniform(0.3, 3),
        "upi_merchant_diversity_score": rng.uniform(0.02, 0.12),
        "utility_bill_payment_consistency": rng.uniform(0.10, 0.45),
        "mobile_recharge_regularity": rng.uniform(25, 50),
        "income_estimate_monthly": max(income, 2_500),
        "income_volatility_cv": rng.uniform(0.50, 0.90),
        "bnpl_outstanding_ratio": rng.uniform(0.30, 0.80),
        "multi_loan_app_count": rng.choice([2, 3, 4, 5]),
        "evening_txn_ratio": rng.uniform(0.35, 0.60),
        "savings_behavior_score": rng.uniform(0.0, 0.10),
        "peer_transfer_reciprocity": rng.uniform(1.5, 3.0),
        "device_tenure_months": rng.uniform(1, 8),
    }


# ─── Persona Registry ────────────────────────────────────────────────────────

PERSONA_GENERATORS = {
    "auto_driver":     (_generate_auto_driver,     0.25),
    "kirana_owner":    (_generate_kirana_owner,     0.20),
    "tuition_teacher": (_generate_tuition_teacher,  0.20),
    "domestic_worker": (_generate_domestic_worker,  0.20),
    "distressed":      (_generate_distressed_profile, 0.15),
}


# ─── Label Engineering ───────────────────────────────────────────────────────

def _compute_approval_label(row: dict, rng: np.random.Generator) -> int:
    """
    Rule-based label with controlled noise.  These rules encode domain
    knowledge about what makes someone creditworthy with alternative data:

    Positive signals:  bill consistency, savings, income stability, device tenure
    Negative signals:  many loan apps, high BNPL ratio, income volatility, low savings
    """
    score = 0.0

    # --- Positive factors ---
    score += row["utility_bill_payment_consistency"] * 25          # max +25
    score += row["savings_behavior_score"] * 15                    # max +15
    score += min(row["upi_txn_frequency_30d"] / 10.0, 1.0) * 10  # max +10
    score += min(row["device_tenure_months"] / 24.0, 1.0) * 10   # max +10
    score += row["upi_merchant_diversity_score"] * 8              # max +8

    if row["income_estimate_monthly"] > 15_000:
        score += 8
    elif row["income_estimate_monthly"] > 10_000:
        score += 4

    if row["peer_transfer_reciprocity"] >= 0.6 and row["peer_transfer_reciprocity"] <= 1.4:
        score += 5  # balanced social transfers = trust signal

    # --- Negative factors ---
    score -= row["income_volatility_cv"] * 15                     # max −15
    score -= row["bnpl_outstanding_ratio"] * 20                   # max −20
    score -= row["multi_loan_app_count"] * 6                      # max −30
    score -= max(row["mobile_recharge_regularity"] - 15, 0) * 0.3  # penalty for irregular

    if row["evening_txn_ratio"] > 0.45:
        score -= 5  # late-night heavy spending can indicate risk

    # Convert score to probability with sigmoid + noise
    prob = 1.0 / (1.0 + np.exp(-0.15 * (score - 30)))
    noise = rng.normal(0, 0.08)
    prob = np.clip(prob + noise, 0.02, 0.98)

    return int(rng.random() < prob)


# ─── Public API ──────────────────────────────────────────────────────────────

def generate_gig_worker_dataset(
    n: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic dataset of gig-worker credit profiles.

    Parameters
    ----------
    n : int
        Number of applicant profiles to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with 12 feature columns + 'persona' + 'loan_approved' target.
    """
    rng = np.random.default_rng(seed)

    # Unpack persona weights
    names = list(PERSONA_GENERATORS.keys())
    generators = [PERSONA_GENERATORS[p][0] for p in names]
    weights = np.array([PERSONA_GENERATORS[p][1] for p in names])
    weights /= weights.sum()  # normalize just in case

    records = []
    for _ in range(n):
        # Pick a persona based on weights
        idx = rng.choice(len(names), p=weights)
        persona_name = names[idx]
        gen_fn = generators[idx]

        profile = gen_fn(rng)
        profile["persona"] = persona_name
        profile[TARGET_COLUMN] = _compute_approval_label(profile, rng)
        records.append(profile)

    df = pd.DataFrame(records)

    # Reorder columns for clarity
    column_order = ["persona"] + FEATURE_COLUMNS + [TARGET_COLUMN]
    df = df[column_order]

    return df


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ALIS RiskMind — Synthetic data generator for gig-worker credit profiles"
    )
    parser.add_argument(
        "--n", type=int, default=5000,
        help="Number of applicant profiles to generate (default: 5000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (default: artifacts/synthetic_gig_workers.csv)",
    )
    args = parser.parse_args()

    # Generate
    df = generate_gig_worker_dataset(n=args.n, seed=args.seed)

    # Save
    out_dir = Path(__file__).parent / "artifacts"
    out_dir.mkdir(exist_ok=True)
    out_path = Path(args.output) if args.output else out_dir / "synthetic_gig_workers.csv"
    df.to_csv(out_path, index=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"  ALIS RiskMind — Synthetic Dataset Generated")
    print(f"{'='*60}")
    print(f"  Samples:        {len(df):,}")
    print(f"  Features:       {len(FEATURE_COLUMNS)}")
    print(f"  Approval rate:  {df[TARGET_COLUMN].mean():.1%}")
    print(f"  Saved to:       {out_path}")
    print(f"\n  Persona distribution:")
    for persona, count in df["persona"].value_counts().items():
        approved = df[df["persona"] == persona][TARGET_COLUMN].mean()
        print(f"    {persona:20s}  n={count:4d}  approval={approved:.1%}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
