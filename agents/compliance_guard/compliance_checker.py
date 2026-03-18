"""
ALIS — ComplianceGuard: Compliance Checker
============================================
Hybrid compliance engine: deterministic rules for numerical thresholds
+ RAG queries for semantic edge cases. Belt and suspenders.

Each rule maps directly to an RBI Digital Lending Guidelines clause.
When a violation is found, the checker returns the exact clause, severity,
and a recommended correction.

Integration with LoanOrchestrator:
  - If is_compliant=True → proceed to ExplainerVoice
  - If severity=CRITICAL  → hard reject, no remediation
  - If severity=HIGH/MED  → Orchestrator attempts auto_adjust_terms()
    and re-runs ComplianceGuard (max 2 retries)

Usage:
    python compliance_checker.py
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rag_engine import RBIQueryEngine

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


# ─── Violation Severity Levels ───────────────────────────────────────────────

class Severity:
    CRITICAL = "CRITICAL"   # hard block — cannot auto-fix
    HIGH = "HIGH"           # must fix before approval, may be auto-adjustable
    MEDIUM = "MEDIUM"       # should fix, loan can proceed with warning
    LOW = "LOW"             # advisory — best practice, not strictly required


# ─── Loan Offer Schema ──────────────────────────────────────────────────────

REQUIRED_FIELDS = {
    "apr": "Annual Percentage Rate (%)",
    "disbursal_account_type": "'own' or 'third_party'",
    "kyc_completed": "Boolean — KYC verified before disbursal",
    "credit_limit_auto_increase": "Boolean — auto credit limit increase enabled",
    "cooling_off_days": "Cooling-off period in days",
    "recovery_contact_hours": "Dict with 'start' and 'end' (24h format)",
}


# ─── Deterministic Rule Engine ───────────────────────────────────────────────

def _check_apr(offer: dict) -> list[dict]:
    """Clause 2.2, 5.2 — APR disclosure and usury check."""
    violations = []
    apr = offer.get("apr")

    if apr is None:
        violations.append({
            "rule": "APR_NOT_DISCLOSED",
            "severity": Severity.CRITICAL,
            "rbi_clause": "2.2",
            "field": "apr",
            "explanation": (
                "Annual Percentage Rate (APR) must be disclosed upfront in the "
                "Key Fact Statement as per RBI Clause 2.2. No APR was provided."
            ),
            "correction": "Compute and disclose APR including all fees and charges.",
        })
        return violations

    if apr > 50:
        violations.append({
            "rule": "APR_EXPLOITATIVE",
            "severity": Severity.CRITICAL,
            "rbi_clause": "5.2",
            "field": "apr",
            "current_value": apr,
            "max_allowed": 50.0,
            "explanation": (
                f"APR of {apr}% exceeds 50% — considered prima facie exploitative "
                f"per RBI Fair Lending Guidelines (Clause 5.2). Regulatory action risk."
            ),
            "correction": f"Reduce APR to ≤50%. Recommended: risk-adjusted rate ≤36%.",
        })
    elif apr > 36:
        violations.append({
            "rule": "APR_REQUIRES_JUSTIFICATION",
            "severity": Severity.HIGH,
            "rbi_clause": "5.2",
            "field": "apr",
            "current_value": apr,
            "threshold": 36.0,
            "explanation": (
                f"APR of {apr}% exceeds 36% for unsecured microloan. Per Clause 5.2, "
                f"additional risk-based justification must be documented in the credit file."
            ),
            "correction": (
                f"Either reduce APR to ≤36% or provide documented risk-based justification "
                f"for the higher rate."
            ),
        })

    return violations


def _check_disbursal(offer: dict) -> list[dict]:
    """Clause 3.1 — Disbursal to borrower's own account only."""
    violations = []
    account_type = offer.get("disbursal_account_type", "").lower()

    if account_type == "third_party":
        violations.append({
            "rule": "THIRD_PARTY_DISBURSAL",
            "severity": Severity.CRITICAL,
            "rbi_clause": "3.1",
            "field": "disbursal_account_type",
            "current_value": account_type,
            "explanation": (
                "Loan disbursal to a third-party account is prohibited. As per "
                "RBI Clause 3.1, all disbursals must be made directly to the "
                "borrower's own bank account. No pass-through or pool accounts."
            ),
            "correction": "Change disbursal to borrower's own bank account.",
        })
    elif account_type not in ("own", "borrower"):
        violations.append({
            "rule": "DISBURSAL_ACCOUNT_UNKNOWN",
            "severity": Severity.HIGH,
            "rbi_clause": "3.1",
            "field": "disbursal_account_type",
            "current_value": account_type,
            "explanation": (
                f"Disbursal account type '{account_type}' is unrecognized. "
                f"RBI Clause 3.1 requires disbursal to borrower's own account."
            ),
            "correction": "Verify and set disbursal_account_type to 'own'.",
        })

    return violations


def _check_kyc(offer: dict) -> list[dict]:
    """Clause 3.2 — KYC must be completed before disbursal."""
    violations = []

    if not offer.get("kyc_completed", False):
        violations.append({
            "rule": "KYC_INCOMPLETE",
            "severity": Severity.CRITICAL,
            "rbi_clause": "3.2",
            "field": "kyc_completed",
            "explanation": (
                "KYC verification not completed. As per RBI Clause 3.2, no loan "
                "shall be disbursed without completion of KYC (Aadhaar eKYC, "
                "Video KYC, or equivalent). PMLA compliance requires this."
            ),
            "correction": "Complete Aadhaar eKYC or Video KYC before proceeding.",
        })

    return violations


def _check_credit_limit(offer: dict) -> list[dict]:
    """Clause 5.1 — No automatic credit limit increase without consent."""
    violations = []

    if offer.get("credit_limit_auto_increase", False):
        violations.append({
            "rule": "AUTO_CREDIT_LIMIT_INCREASE",
            "severity": Severity.HIGH,
            "rbi_clause": "5.1",
            "field": "credit_limit_auto_increase",
            "explanation": (
                "Automatic credit limit increase is enabled without explicit "
                "borrower consent. RBI Clause 5.1 prohibits this for all digital "
                "lending products including BNPL."
            ),
            "correction": "Disable auto-increase. Require explicit borrower consent for any limit change.",
        })

    return violations


def _check_cooling_off(offer: dict) -> list[dict]:
    """Clause 2.3 — Mandatory cooling-off period."""
    violations = []
    cooling_off = offer.get("cooling_off_days")

    if cooling_off is None:
        violations.append({
            "rule": "COOLING_OFF_NOT_SPECIFIED",
            "severity": Severity.HIGH,
            "rbi_clause": "2.3",
            "field": "cooling_off_days",
            "explanation": (
                "Cooling-off/look-up period not specified. RBI Clause 2.3 mandates "
                "a minimum 3-day cooling-off period for loans ≥7 days tenure. "
                "No penalty may be charged during this period."
            ),
            "correction": "Set cooling_off_days ≥ 3 for loans with tenure ≥ 7 days.",
        })
    elif cooling_off < 3:
        violations.append({
            "rule": "COOLING_OFF_TOO_SHORT",
            "severity": Severity.HIGH,
            "rbi_clause": "2.3",
            "field": "cooling_off_days",
            "current_value": cooling_off,
            "min_required": 3,
            "explanation": (
                f"Cooling-off period of {cooling_off} day(s) is below the RBI "
                f"minimum of 3 days (Clause 2.3). During cooling-off, no penalty "
                f"or charges may apply on prepayment."
            ),
            "correction": f"Increase cooling_off_days to at least 3.",
        })

    return violations


def _check_recovery_hours(offer: dict) -> list[dict]:
    """Clause 6.1 — Recovery contact only between 8 AM–8 PM."""
    violations = []
    hours = offer.get("recovery_contact_hours", {})
    start = hours.get("start")
    end = hours.get("end")

    if start is None or end is None:
        violations.append({
            "rule": "RECOVERY_HOURS_NOT_SPECIFIED",
            "severity": Severity.MEDIUM,
            "rbi_clause": "6.1",
            "field": "recovery_contact_hours",
            "explanation": (
                "Recovery contact hours not specified. RBI Clause 6.1 restricts "
                "recovery communication to 8:00 AM – 8:00 PM."
            ),
            "correction": "Set recovery_contact_hours: {start: 8, end: 20}.",
        })
    else:
        if start < 8:
            violations.append({
                "rule": "RECOVERY_BEFORE_8AM",
                "severity": Severity.HIGH,
                "rbi_clause": "6.1",
                "field": "recovery_contact_hours",
                "current_value": f"{start}:00",
                "explanation": (
                    f"Recovery contact start time {start}:00 is before 8:00 AM. "
                    f"RBI Clause 6.1 prohibits contact before 8 AM."
                ),
                "correction": "Set recovery start hour to 8 (8:00 AM).",
            })
        if end > 20:
            violations.append({
                "rule": "RECOVERY_AFTER_8PM",
                "severity": Severity.HIGH,
                "rbi_clause": "6.1",
                "field": "recovery_contact_hours",
                "current_value": f"{end}:00",
                "explanation": (
                    f"Recovery contact end time {end}:00 is after 8:00 PM. "
                    f"RBI Clause 6.1 prohibits contact after 8 PM."
                ),
                "correction": "Set recovery end hour to 20 (8:00 PM).",
            })

    return violations


# ─── Compliance Check Rules Registry ────────────────────────────────────────

RULE_CHECKS = [
    _check_apr,
    _check_disbursal,
    _check_kyc,
    _check_credit_limit,
    _check_cooling_off,
    _check_recovery_hours,
]


# ─── Main Compliance Checker ────────────────────────────────────────────────

def check_loan_compliance(
    loan_offer: dict,
    use_rag: bool = True,
) -> dict:
    """
    Check a loan offer against RBI Digital Lending Guidelines.

    Runs deterministic rule checks first, then optionally queries
    the RAG engine for any edge-case semantic checks.

    Parameters
    ----------
    loan_offer : dict
        Loan offer with fields matching REQUIRED_FIELDS.
    use_rag : bool
        Whether to also run RAG-based semantic checks.

    Returns
    -------
    dict with:
        is_compliant: bool
        violations: list of violation dicts
        compliance_report: str (formatted report)
        recommended_corrections: list of str
        rag_context: list of retrieved RBI clauses (if RAG enabled)
        offer_hash: str (SHA-256 of the offer for audit)
        timestamp: str (ISO 8601)
    """
    violations = []

    # ── 1. Run deterministic rule checks ─────────────────────────────────────
    for check_fn in RULE_CHECKS:
        violations.extend(check_fn(loan_offer))

    # ── 2. RAG-based semantic check (edge cases) ────────────────────────────
    rag_context = []
    if use_rag:
        try:
            engine = RBIQueryEngine()
            if engine.available:
                # Query for relevant clauses based on the loan offer
                apr = loan_offer.get("apr", "not specified")
                query = (
                    f"Are there any regulatory requirements for a digital loan with "
                    f"APR {apr}% disbursed to {loan_offer.get('disbursal_account_type', 'unknown')} "
                    f"account with cooling off period of "
                    f"{loan_offer.get('cooling_off_days', 'not specified')} days?"
                )
                rag_context = engine.query(query, n_results=3)
        except Exception:
            pass  # RAG is supplementary, not critical

    # ── 3. Compile results ───────────────────────────────────────────────────
    has_critical = any(v["severity"] == Severity.CRITICAL for v in violations)
    has_high = any(v["severity"] == Severity.HIGH for v in violations)
    is_compliant = len(violations) == 0

    # Corrections
    corrections = [v["correction"] for v in violations if "correction" in v]

    # Formatted report
    report = _format_compliance_report(loan_offer, violations, is_compliant)

    # Offer hash for audit
    offer_str = json.dumps(loan_offer, sort_keys=True, default=str)
    offer_hash = hashlib.sha256(offer_str.encode()).hexdigest()[:16]

    return {
        "is_compliant": is_compliant,
        "has_critical_violations": has_critical,
        "has_high_violations": has_high,
        "violation_count": len(violations),
        "violations": violations,
        "compliance_report": report,
        "recommended_corrections": corrections,
        "rag_context": rag_context,
        "offer_hash": offer_hash,
        "timestamp": datetime.utcnow().isoformat(),
    }


def _format_compliance_report(
    offer: dict, violations: list, is_compliant: bool
) -> str:
    """Generate a formatted compliance report string."""
    lines = [
        "╔══════════════════════════════════════════════════════════╗",
        "║    ALIS ComplianceGuard — RBI Compliance Report         ║",
        "╚══════════════════════════════════════════════════════════╝",
        "",
        f"  Status:     {'✅ COMPLIANT' if is_compliant else '❌ NON-COMPLIANT'}",
        f"  Violations: {len(violations)}",
        f"  Timestamp:  {datetime.utcnow().isoformat()}",
        "",
        "  ── Loan Offer Summary ──────────────────────────────────",
    ]

    for key, desc in REQUIRED_FIELDS.items():
        val = offer.get(key, "NOT PROVIDED")
        lines.append(f"    {desc:45s} {val}")

    if violations:
        lines.append("")
        lines.append("  ── Violations Found ────────────────────────────────────")
        for i, v in enumerate(violations, 1):
            severity_icon = {
                "CRITICAL": "🚫", "HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵"
            }.get(v["severity"], "⚪")

            lines.append(f"")
            lines.append(f"    {severity_icon} [{i}] {v['rule']} ({v['severity']})")
            lines.append(f"       RBI Clause: {v['rbi_clause']}")
            lines.append(f"       {v['explanation']}")
            if "correction" in v:
                lines.append(f"       Fix: {v['correction']}")
    else:
        lines.append("")
        lines.append("  ✅ All checks passed. Loan terms comply with RBI guidelines.")

    lines.append("")
    lines.append("  ═══════════════════════════════════════════════════════")
    return "\n".join(lines)


# ─── Auto-Correction (used by LoanOrchestrator) ─────────────────────────────

def auto_adjust_terms(
    loan_offer: dict,
    violations: list[dict],
) -> dict:
    """
    Attempt to auto-correct fixable violations.

    This is called by the LoanOrchestrator when ComplianceGuard returns
    non-critical violations. The Orchestrator adjusts terms and re-runs
    compliance check (max 2 retries).

    Only adjusts:
    - APR (cap at 36%)
    - Cooling-off days (set to 3)
    - Recovery hours (set to 8-20)
    - Credit limit auto-increase (disable)

    Does NOT adjust (requires human decision):
    - Disbursal account type
    - KYC status
    """
    adjusted = loan_offer.copy()
    adjustments_made = []

    for v in violations:
        if v["severity"] == Severity.CRITICAL:
            continue  # cannot auto-fix critical

        field = v.get("field")

        if field == "apr" and v["rule"] == "APR_REQUIRES_JUSTIFICATION":
            adjusted["apr"] = min(adjusted.get("apr", 100), 36.0)
            adjustments_made.append(f"APR reduced to 36%")

        elif field == "cooling_off_days":
            adjusted["cooling_off_days"] = max(adjusted.get("cooling_off_days", 0), 3)
            adjustments_made.append(f"Cooling-off increased to 3 days")

        elif field == "recovery_contact_hours":
            hours = adjusted.get("recovery_contact_hours", {})
            hours["start"] = max(hours.get("start", 8), 8)
            hours["end"] = min(hours.get("end", 20), 20)
            adjusted["recovery_contact_hours"] = hours
            adjustments_made.append(f"Recovery hours adjusted to 8AM-8PM")

        elif field == "credit_limit_auto_increase":
            adjusted["credit_limit_auto_increase"] = False
            adjustments_made.append("Auto credit limit increase disabled")

    adjusted["_adjustments_made"] = adjustments_made
    return adjusted


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print("  ALIS ComplianceGuard — Compliance Checker Demo")
    print(f"{'='*60}\n")

    # ── Test Case 1: Compliant offer ─────────────────────────────────────────
    print("  ── Test 1: Compliant Loan Offer ──\n")
    compliant_offer = {
        "apr": 24.0,
        "disbursal_account_type": "own",
        "kyc_completed": True,
        "credit_limit_auto_increase": False,
        "cooling_off_days": 5,
        "recovery_contact_hours": {"start": 9, "end": 18},
    }
    result = check_loan_compliance(compliant_offer, use_rag=False)
    print(result["compliance_report"])

    # ── Test Case 2: Multiple violations ─────────────────────────────────────
    print("\n  ── Test 2: Non-Compliant Loan Offer ──\n")
    bad_offer = {
        "apr": 55.0,
        "disbursal_account_type": "third_party",
        "kyc_completed": False,
        "credit_limit_auto_increase": True,
        "cooling_off_days": 1,
        "recovery_contact_hours": {"start": 6, "end": 22},
    }
    result = check_loan_compliance(bad_offer, use_rag=False)
    print(result["compliance_report"])

    print(f"\n  Violations: {result['violation_count']}")
    print(f"  Critical:   {result['has_critical_violations']}")

    # ── Test Case 3: Auto-correction ─────────────────────────────────────────
    print("\n  ── Test 3: Auto-Correction Demo ──\n")
    fixable_offer = {
        "apr": 38.0,
        "disbursal_account_type": "own",
        "kyc_completed": True,
        "credit_limit_auto_increase": True,
        "cooling_off_days": 2,
        "recovery_contact_hours": {"start": 7, "end": 21},
    }
    result = check_loan_compliance(fixable_offer, use_rag=False)
    print(f"  Before: {result['violation_count']} violations")

    adjusted = auto_adjust_terms(fixable_offer, result["violations"])
    result2 = check_loan_compliance(adjusted, use_rag=False)
    print(f"  After auto-adjust: {result2['violation_count']} violations")
    print(f"  Adjustments: {adjusted.get('_adjustments_made', [])}")
    print(result2["compliance_report"])

    print(f"\n  ✓ Done.\n")


if __name__ == "__main__":
    main()
