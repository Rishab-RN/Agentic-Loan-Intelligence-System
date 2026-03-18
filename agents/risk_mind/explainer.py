"""
ALIS — RiskMind: SHAP Explanation Engine
==========================================
Per-applicant credit score explanations using SHAP TreeExplainer.
Every decision includes: credit score (0-900), top positive/negative
factors, counterfactual advice, and optional SHAP waterfall plots.

Usage:
    python explainer.py           # explain a sample applicant
    python explainer.py --plot    # also save SHAP waterfall plot
"""

import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from data_generator import FEATURE_COLUMNS, FEATURE_DESCRIPTIONS

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
APPROVAL_THRESHOLD = 500  # score >= 500 → approved


class RiskMindExplainer:
    """
    Wraps the trained XGBoost model + SHAP for production-grade explanations.

    We use the *base* (uncalibrated) model for SHAP (TreeExplainer needs native
    trees) and the *calibrated* model for the actual credit score probability.
    """

    def __init__(self, model_path=None, base_model_path=None, scaler_path=None):
        model_path = Path(model_path or ARTIFACTS_DIR / "riskmind_model.joblib")
        base_model_path = Path(base_model_path or ARTIFACTS_DIR / "riskmind_base_model.joblib")
        scaler_path = Path(scaler_path or ARTIFACTS_DIR / "riskmind_scaler.joblib")

        self.calibrated_model = joblib.load(model_path)
        self.base_model = joblib.load(base_model_path)
        self.scaler = joblib.load(scaler_path)
        self.shap_explainer = shap.TreeExplainer(self.base_model)

    def _probability_to_score(self, prob: float) -> int:
        """Convert approval probability [0,1] → credit score [0,900]."""
        return int(np.clip(prob, 0.0, 1.0) * 900)

    def explain_decision(self, applicant_features: dict) -> dict:
        """
        Full credit decision explanation for one applicant.

        Returns dict with: credit_score, approved, probability,
        top_positive_factors, top_negative_factors, counterfactual_advice,
        shap_values.
        """
        missing = set(FEATURE_COLUMNS) - set(applicant_features.keys())
        if missing:
            raise ValueError(f"Missing features: {missing}")

        # Build feature vector
        feature_values = [applicant_features[f] for f in FEATURE_COLUMNS]
        X_raw = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_raw), columns=FEATURE_COLUMNS,
        )

        # Credit score from calibrated model
        probability = float(self.calibrated_model.predict_proba(X_scaled)[0, 1])
        credit_score = self._probability_to_score(probability)
        approved = credit_score >= APPROVAL_THRESHOLD

        # SHAP values from base model
        shap_values_array = self.shap_explainer.shap_values(X_scaled)
        if isinstance(shap_values_array, list):
            sv = shap_values_array[1][0]
        else:
            sv = shap_values_array[0]

        shap_map = {
            FEATURE_COLUMNS[i]: float(sv[i]) for i in range(len(FEATURE_COLUMNS))
        }

        sorted_features = sorted(shap_map.items(), key=lambda x: x[1], reverse=True)

        # Top 5 positive factors
        top_positive = []
        for feat, val in sorted_features:
            if val > 0 and len(top_positive) < 5:
                top_positive.append({
                    "feature": feat,
                    "description": FEATURE_DESCRIPTIONS.get(feat, feat),
                    "value": float(applicant_features[feat]),
                    "shap_impact": round(val, 4),
                    "explanation": self._explain_factor(feat, applicant_features[feat], "positive"),
                })

        # Top 3 negative factors
        top_negative = []
        for feat, val in reversed(sorted_features):
            if val < 0 and len(top_negative) < 3:
                top_negative.append({
                    "feature": feat,
                    "description": FEATURE_DESCRIPTIONS.get(feat, feat),
                    "value": float(applicant_features[feat]),
                    "shap_impact": round(val, 4),
                    "explanation": self._explain_factor(feat, applicant_features[feat], "negative"),
                })

        counterfactual = self._generate_counterfactuals(
            applicant_features, shap_map, credit_score
        )

        return {
            "credit_score": credit_score,
            "approved": approved,
            "approval_threshold": APPROVAL_THRESHOLD,
            "probability": round(probability, 4),
            "top_positive_factors": top_positive,
            "top_negative_factors": top_negative,
            "counterfactual_advice": counterfactual,
            "shap_values": shap_map,
        }

    def _explain_factor(self, feature: str, value: float, direction: str) -> str:
        """Generate a plain-English explanation for a single factor."""
        templates = {
            "utility_bill_payment_consistency": {
                "positive": f"You paid {value:.0%} of utility bills on time — strong financial discipline.",
                "negative": f"Only {value:.0%} of utility bills paid on time — consistent payment is the strongest credit signal.",
            },
            "savings_behavior_score": {
                "positive": f"Savings regularity of {value:.2f} shows you consistently set money aside.",
                "negative": f"Savings score of {value:.2f} is low — regular savings demonstrate financial planning.",
            },
            "upi_txn_frequency_30d": {
                "positive": f"You average {value:.1f} UPI txns/day, showing active economic participation.",
                "negative": f"Only {value:.1f} UPI txns/day — higher digital activity strengthens your profile.",
            },
            "income_estimate_monthly": {
                "positive": f"Estimated monthly income of ₹{value:,.0f} supports the requested loan amount.",
                "negative": f"Estimated monthly income of ₹{value:,.0f} is below the comfort zone for this loan.",
            },
            "income_volatility_cv": {
                "positive": f"Income volatility of {value:.2f} indicates stable earnings.",
                "negative": f"Income volatility of {value:.2f} is high — lenders prefer predictable income.",
            },
            "bnpl_outstanding_ratio": {
                "positive": f"BNPL-to-income ratio of {value:.2f} shows responsible BNPL usage.",
                "negative": f"BNPL-to-income ratio of {value:.2f} is elevated — reduce outstanding BNPL balances.",
            },
            "multi_loan_app_count": {
                "positive": f"Having {int(value)} loan app(s) is within normal range.",
                "negative": f"Having {int(value)} loan apps installed signals financial distress.",
            },
            "device_tenure_months": {
                "positive": f"Device used for {value:.0f} months — stability is a positive signal.",
                "negative": f"Device tenure of only {value:.0f} months is short — longer usage shows stability.",
            },
            "upi_merchant_diversity_score": {
                "positive": f"Merchant diversity of {value:.2f} shows varied, genuine spending.",
                "negative": f"Merchant diversity of {value:.2f} is low — diverse spending strengthens your profile.",
            },
            "mobile_recharge_regularity": {
                "positive": f"Recharging every {value:.0f} days shows consistent phone usage.",
                "negative": f"Irregular recharges (every {value:.0f} days) suggest inconsistent engagement.",
            },
            "evening_txn_ratio": {
                "positive": f"Evening transaction ratio of {value:.2f} is within healthy range.",
                "negative": f"High evening transaction ratio ({value:.2f}) can indicate risky spending.",
            },
            "peer_transfer_reciprocity": {
                "positive": f"Peer transfer ratio of {value:.2f} shows balanced financial relationships.",
                "negative": f"Peer transfer ratio of {value:.2f} is skewed — balanced patterns build trust.",
            },
        }

        if feature in templates and direction in templates[feature]:
            return templates[feature][direction]

        desc = FEATURE_DESCRIPTIONS.get(feature, feature)
        verb = "helping" if direction == "positive" else "reducing"
        return f"{desc}: value of {value:.2f} is {verb} your score."

    def _generate_counterfactuals(self, features, shap_map, current_score):
        """Generate actionable counterfactual advice from negative SHAP factors."""
        targets = {
            "utility_bill_payment_consistency": (0.90, "Pay electricity/water bills on time for 3 consecutive months"),
            "savings_behavior_score": (0.50, "Set aside even ₹500/month into savings regularly"),
            "bnpl_outstanding_ratio": (0.10, "Pay down BNPL balances to below 10% of monthly income"),
            "multi_loan_app_count": (1, "Uninstall unused loan apps — many installed signals distress"),
            "income_volatility_cv": (0.25, "Diversify income sources to reduce month-to-month variation"),
            "mobile_recharge_regularity": (10, "Recharge phone at regular intervals (every 10-15 days)"),
            "device_tenure_months": (18, "Keep using your current device — stability builds over time"),
        }

        lower_is_better = {"mobile_recharge_regularity", "income_volatility_cv",
                           "bnpl_outstanding_ratio", "multi_loan_app_count"}

        negative_feats = sorted(
            [(k, v) for k, v in shap_map.items() if v < 0], key=lambda x: x[1]
        )

        counterfactuals = []
        for feat, shap_val in negative_feats:
            if feat not in targets:
                continue
            target_val, advice = targets[feat]
            current_val = features[feat]

            if feat in lower_is_better and current_val <= target_val:
                continue
            if feat not in lower_is_better and current_val >= target_val:
                continue

            improvement = max(int(abs(shap_val) * 900 * 0.25), 10)
            counterfactuals.append({
                "feature": feat,
                "current_value": round(current_val, 2),
                "target_value": target_val,
                "advice": advice,
                "estimated_score_improvement": improvement,
                "projected_score": min(current_score + improvement, 900),
            })
            if len(counterfactuals) >= 3:
                break

        return counterfactuals

    def generate_waterfall_plot(self, applicant_features, save_path=None):
        """Generate and save a SHAP waterfall plot for one applicant."""
        feature_values = [applicant_features[f] for f in FEATURE_COLUMNS]
        X_raw = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_raw), columns=FEATURE_COLUMNS,
        )

        explanation = self.shap_explainer(X_scaled)
        explanation.feature_names = [
            FEATURE_DESCRIPTIONS.get(f, f).split("(")[0].strip()
            for f in FEATURE_COLUMNS
        ]

        fig, ax = plt.subplots(figsize=(12, 8))
        plt.sca(ax)
        shap.plots.waterfall(explanation[0], max_display=12, show=False)
        plt.title("ALIS RiskMind — SHAP Credit Decision Breakdown", fontsize=14, pad=20)
        plt.tight_layout()

        save_path = Path(save_path or ARTIFACTS_DIR / "shap_waterfall.png")
        save_path.parent.mkdir(exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  SHAP waterfall plot saved: {save_path}")
        return save_path


# ─── CLI Demo ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ALIS RiskMind — Explanation demo")
    parser.add_argument("--plot", action="store_true", help="Save SHAP waterfall plot")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  ALIS RiskMind — Explanation Engine Demo")
    print(f"{'='*60}\n")

    explainer = RiskMindExplainer()

    # Sample applicant: kirana shop owner in Ballari
    sample = {
        "upi_txn_frequency_30d": 12.5,
        "upi_merchant_diversity_score": 0.18,
        "utility_bill_payment_consistency": 0.78,
        "mobile_recharge_regularity": 12.0,
        "income_estimate_monthly": 22000.0,
        "income_volatility_cv": 0.28,
        "bnpl_outstanding_ratio": 0.12,
        "multi_loan_app_count": 1,
        "evening_txn_ratio": 0.25,
        "savings_behavior_score": 0.45,
        "peer_transfer_reciprocity": 0.75,
        "device_tenure_months": 24.0,
    }

    print("  Sample: Kirana shop owner, Ballari")
    for feat, val in sample.items():
        desc = FEATURE_DESCRIPTIONS.get(feat, feat).split("(")[0].strip()
        print(f"    {desc:45s} {val}")

    result = explainer.explain_decision(sample)

    print(f"\n  {'═'*50}")
    print(f"  CREDIT SCORE:  {result['credit_score']} / 900")
    print(f"  DECISION:      {'✓ APPROVED' if result['approved'] else '✗ REJECTED'}")
    print(f"  PROBABILITY:   {result['probability']:.2%}")
    print(f"  {'═'*50}")

    print(f"\n  ✓ Top Positive Factors:")
    for f in result["top_positive_factors"]:
        print(f"    [+{f['shap_impact']:+.3f}]  {f['explanation']}")

    print(f"\n  ✗ Top Negative Factors:")
    for f in result["top_negative_factors"]:
        print(f"    [{f['shap_impact']:+.3f}]  {f['explanation']}")

    if result["counterfactual_advice"]:
        print(f"\n  ℹ Counterfactual Advice:")
        for cf in result["counterfactual_advice"]:
            print(f"    → {cf['advice']}")
            print(f"      (+{cf['estimated_score_improvement']} pts → {cf['projected_score']})")

    if args.plot:
        print()
        explainer.generate_waterfall_plot(sample)

    print(f"\n  ✓ Done.\n")


if __name__ == "__main__":
    main()
