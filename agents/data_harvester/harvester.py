"""
ALIS — DataHarvester: Application Data Validator & Feature Engineer
====================================================================
Agent 1 in the ALIS pipeline. Responsible for:
  1. Schema validation — ensures all required fields are present
  2. Feature normalization — scales raw inputs into model-ready ranges
  3. Thin-file detection — activates gamified observation protocol
  4. Data enrichment — computes derived features from raw UPI data

The DataHarvester is the gatekeeper: no garbage in → no garbage out.
If an application fails validation, the pipeline halts early with
a clear error message.

Usage:
    from harvester import DataHarvester
    dh = DataHarvester()
    result = dh.harvest(application_data)
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("DataHarvester")


# ─── Feature Schema (what RiskMind's XGBoost model expects) ─────────────────

RISKMIND_FEATURES = [
    "upi_txn_frequency_30d",
    "upi_avg_txn_amount",
    "upi_merchant_diversity_score",
    "utility_bill_payment_consistency",
    "mobile_recharge_regularity",
    "savings_behavior_score",
    "income_estimate_monthly",
    "income_volatility_cv",
    "bnpl_outstanding_ratio",
    "multi_loan_app_count",
    "peer_transfer_reciprocity",
    "evening_txn_ratio",
    "device_tenure_months",
]

# Required fields that must be present in every application
REQUIRED_FIELDS = {
    "applicant_name": str,
    "loan_amount": (int, float),
}

# Optional but important fields with defaults
OPTIONAL_DEFAULTS = {
    "upi_history_days": 180,
    "upi_txn_frequency_30d": 45,
    "upi_avg_txn_amount": 250,
    "upi_merchant_diversity_score": 0.4,
    "utility_bill_payment_consistency": 0.7,
    "mobile_recharge_regularity": 0.6,
    "savings_behavior_score": 40,
    "income_estimate_monthly": 15000,
    "income_volatility_cv": 0.35,
    "bnpl_outstanding_ratio": 0.15,
    "multi_loan_app_count": 2,
    "peer_transfer_reciprocity": 0.5,
    "evening_txn_ratio": 0.15,
    "device_tenure_months": 18,
    "kyc_completed": True,
    "disbursal_account_type": "own",
}

# Thin-file threshold
THIN_FILE_THRESHOLD_DAYS = 30


# ─── Validation Rules ───────────────────────────────────────────────────────

FIELD_RANGES = {
    "loan_amount": (1000, 500000),
    "upi_txn_frequency_30d": (0, 1000),
    "upi_avg_txn_amount": (1, 100000),
    "upi_merchant_diversity_score": (0.0, 1.0),
    "utility_bill_payment_consistency": (0.0, 1.0),
    "mobile_recharge_regularity": (0.0, 1.0),
    "savings_behavior_score": (0, 100),
    "income_estimate_monthly": (0, 1000000),
    "income_volatility_cv": (0.0, 3.0),
    "bnpl_outstanding_ratio": (0.0, 5.0),
    "multi_loan_app_count": (0, 20),
    "peer_transfer_reciprocity": (0.0, 1.0),
    "evening_txn_ratio": (0.0, 1.0),
    "device_tenure_months": (0, 120),
    "upi_history_days": (0, 3650),
}


class DataHarvester:
    """
    Validates, normalizes, and enriches loan application data.

    This is the first agent in the ALIS pipeline. It ensures that
    downstream agents (FraudSentinel, RiskMind, etc.) receive
    clean, well-structured data.
    """

    def harvest(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point. Validate, normalize, and enrich application data.

        Parameters
        ----------
        application_data : dict
            Raw application data from the borrower/API.

        Returns
        -------
        dict with keys:
            harvested_data: dict — cleaned and enriched data
            is_thin_file: bool — whether thin-file protocol is activated
            validation_errors: list — any non-fatal validation warnings
            feature_vector: dict — model-ready feature vector for RiskMind
        """
        errors = []
        warnings = []

        # ── 1. Schema validation ─────────────────────────────────────────
        data = application_data.copy()
        is_valid, validation_msgs = self._validate_schema(data)
        if not is_valid:
            errors.extend(validation_msgs)
        else:
            warnings.extend(validation_msgs)

        # ── 2. Fill defaults for missing optional fields ─────────────────
        data = self._fill_defaults(data)

        # ── 3. Clamp values to valid ranges ──────────────────────────────
        data = self._clamp_ranges(data)

        # ── 4. Thin-file detection ───────────────────────────────────────
        is_thin_file = self._detect_thin_file(data)
        if is_thin_file:
            logger.warning(
                f"   [THIN-FILE PROTOCOL] Applicant has only "
                f"{data.get('upi_history_days', 0)} days of history. "
                f"Activating 30-day gamified observation period."
            )

        # ── 5. Feature enrichment ────────────────────────────────────────
        data = self._enrich_features(data)

        # ── 6. Extract feature vector for RiskMind ───────────────────────
        feature_vector = self._extract_feature_vector(data)

        # ── 7. Compute applicant risk tier (for FraudSentinel routing) ───
        risk_tier = self._compute_initial_risk_tier(data)
        data["_initial_risk_tier"] = risk_tier

        return {
            "harvested_data": data,
            "is_thin_file": is_thin_file,
            "validation_errors": errors,
            "validation_warnings": warnings,
            "feature_vector": feature_vector,
        }

    def _validate_schema(
        self, data: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate that required fields are present and have correct types."""
        messages = []
        is_valid = True

        for field, expected_type in REQUIRED_FIELDS.items():
            if field not in data:
                messages.append(f"Missing required field: '{field}'")
                is_valid = False
            elif not isinstance(data[field], expected_type):
                messages.append(
                    f"Field '{field}' has type {type(data[field]).__name__}, "
                    f"expected {expected_type.__name__ if isinstance(expected_type, type) else expected_type}"
                )

        # Warnings for missing optional fields
        for field in OPTIONAL_DEFAULTS:
            if field not in data:
                messages.append(
                    f"Optional field '{field}' not provided, using default: "
                    f"{OPTIONAL_DEFAULTS[field]}"
                )

        return is_valid, messages

    def _fill_defaults(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fill in default values for missing optional fields."""
        for field, default in OPTIONAL_DEFAULTS.items():
            if field not in data:
                data[field] = default
        return data

    def _clamp_ranges(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clamp numerical values to valid ranges."""
        for field, (lo, hi) in FIELD_RANGES.items():
            if field in data and isinstance(data[field], (int, float)):
                original = data[field]
                data[field] = max(lo, min(hi, data[field]))
                if data[field] != original:
                    logger.info(
                        f"   Clamped '{field}': {original} → {data[field]} "
                        f"(valid range: [{lo}, {hi}])"
                    )
        return data

    def _detect_thin_file(self, data: Dict[str, Any]) -> bool:
        """Detect if the applicant is a 'thin file' (insufficient history)."""
        history_days = data.get("upi_history_days", 0)
        return history_days < THIN_FILE_THRESHOLD_DAYS

    def _enrich_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute derived features from raw data.

        These are features that RiskMind expects but may not be
        directly provided in the application — they're computed
        from other raw inputs.
        """
        # Debt-to-income ratio (approximate)
        income = data.get("income_estimate_monthly", 15000)
        loan_amount = data.get("loan_amount", 10000)
        if income > 0:
            data["debt_to_income_ratio"] = round(
                (loan_amount / 12) / income, 4
            )
        else:
            data["debt_to_income_ratio"] = 1.0

        # Digital activity score (composite)
        upi_freq = data.get("upi_txn_frequency_30d", 0)
        merchant_div = data.get("upi_merchant_diversity_score", 0)
        recharge_reg = data.get("mobile_recharge_regularity", 0)
        data["digital_activity_score"] = round(
            (min(upi_freq / 100, 1.0) * 0.4
             + merchant_div * 0.3
             + recharge_reg * 0.3),
            4,
        )

        # Financial discipline score (composite)
        utility = data.get("utility_bill_payment_consistency", 0)
        savings = data.get("savings_behavior_score", 0) / 100
        bnpl = data.get("bnpl_outstanding_ratio", 0)
        data["financial_discipline_score"] = round(
            utility * 0.4 + savings * 0.35 + max(0, 1 - bnpl) * 0.25,
            4,
        )

        return data

    def _extract_feature_vector(
        self, data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract the exact feature vector that RiskMind XGBoost expects."""
        return {
            feat: float(data.get(feat, OPTIONAL_DEFAULTS.get(feat, 0)))
            for feat in RISKMIND_FEATURES
        }

    def _compute_initial_risk_tier(self, data: Dict[str, Any]) -> str:
        """
        Quick heuristic risk tier for FraudSentinel prioritization.

        Not a credit decision — just a triage flag.
        """
        loan_amount = data.get("loan_amount", 0)
        evening_ratio = data.get("evening_txn_ratio", 0)
        loan_apps = data.get("multi_loan_app_count", 0)

        red_flags = 0
        if loan_amount > 100000:
            red_flags += 1
        if evening_ratio > 0.5:
            red_flags += 1
        if loan_apps > 5:
            red_flags += 1

        if red_flags >= 2:
            return "HIGH"
        elif red_flags == 1:
            return "MEDIUM"
        return "LOW"


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO)

    print(f"\n{'='*60}")
    print("  ALIS DataHarvester — Validation & Enrichment Demo")
    print(f"{'='*60}\n")

    harvester = DataHarvester()

    # Test 1: Full application
    print("  ── Test 1: Full Application (Ramu) ──\n")
    ramu_app = {
        "applicant_name": "Ramu",
        "loan_amount": 25000,
        "upi_history_days": 350,
        "upi_txn_frequency_30d": 85,
        "upi_avg_txn_amount": 180,
        "upi_merchant_diversity_score": 0.65,
        "utility_bill_payment_consistency": 0.92,
        "mobile_recharge_regularity": 0.88,
        "savings_behavior_score": 55,
        "income_estimate_monthly": 18000,
        "income_volatility_cv": 0.28,
        "bnpl_outstanding_ratio": 0.10,
        "multi_loan_app_count": 1,
        "peer_transfer_reciprocity": 0.62,
        "evening_txn_ratio": 0.12,
        "device_tenure_months": 24,
    }
    result = harvester.harvest(ramu_app)
    print(f"  Valid: {len(result['validation_errors']) == 0}")
    print(f"  Thin File: {result['is_thin_file']}")
    print(f"  Risk Tier: {result['harvested_data']['_initial_risk_tier']}")
    print(f"  Digital Activity Score: {result['harvested_data']['digital_activity_score']}")
    print(f"  Financial Discipline: {result['harvested_data']['financial_discipline_score']}")
    print(f"  Feature Vector Keys: {list(result['feature_vector'].keys())}")

    # Test 2: Thin-file
    print("\n  ── Test 2: Thin-File Application ──\n")
    thin_app = {
        "applicant_name": "New User",
        "loan_amount": 5000,
        "upi_history_days": 12,
    }
    result2 = harvester.harvest(thin_app)
    print(f"  Thin File: {result2['is_thin_file']}")
    print(f"  Warnings: {len(result2['validation_warnings'])} defaults applied")

    # Test 3: Missing required fields
    print("\n  ── Test 3: Invalid Application ──\n")
    bad_app = {"loan_amount": -500}
    result3 = harvester.harvest(bad_app)
    print(f"  Errors: {result3['validation_errors']}")

    print(f"\n  ✓ Done.\n")


if __name__ == "__main__":
    main()
