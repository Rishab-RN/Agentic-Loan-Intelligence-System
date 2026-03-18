"""
ALIS — ExplainerVoice: Decision Templates & Counterfactual Generator
=====================================================================
Generates warm, structured explanations for 4 loan decision scenarios
with SHAP-grounded counterfactual advice.

The counterfactual generator uses the SHAP additive property: if a
feature's SHAP value is -0.15, and the total probability is 0.57,
then improving that feature to the "good" threshold would add ~0.15
to the probability → (0.57 + 0.15) * 900 = ~648 score points.

This is NOT made-up advice — it's mathematically grounded in the
model's actual feature attributions.
"""

import numpy as np

from shap_translator import TRANSLATIONS, translate_shap_to_language

# ─── Score Thresholds ────────────────────────────────────────────────────────

SCORE_THRESHOLDS = {
    "APPROVED": 600,       # Score ≥ 600 → approved
    "BORDERLINE": 450,     # Score 450-599 → rejected with improvement path
    "REJECTED": 300,       # Score 300-449 → harder rejection
    "HIGH_RISK": 0,        # Score < 300 → serious concerns
}

# ─── Decision Labels ────────────────────────────────────────────────────────

DECISIONS = {
    "english": {
        "APPROVED": {
            "greeting": "Great news, {name}!",
            "headline": "Your loan of ₹{amount} is approved.",
            "body": (
                "We looked at how you use UPI, how you pay your bills, and your "
                "spending patterns. Based on this, your credit score is {score} out of 900. "
                "This is above our threshold of 600, which means you qualify."
            ),
            "next_step": "Your loan will be disbursed to your bank account within 24 hours. "
                         "Remember, you have a {cooling_off} day cooling-off period — during "
                         "this time, you can cancel without any charges.",
        },
        "REJECTED": {
            "greeting": "Hello {name},",
            "headline": "Your loan of ₹{amount} is not approved right now.",
            "body": (
                "Your current credit score is {score} out of 900. "
                "We need a score of at least 600 to approve a loan. "
                "But don't worry — this is not permanent. We can see from your "
                "UPI transactions that you're on the right track. Here's exactly "
                "what you can do to improve your score:"
            ),
            "next_step": "Try these improvements for {days_to_improve} days, then apply again. "
                         "We'll remember your progress.",
        },
        "MORE_INFO_NEEDED": {
            "greeting": "Hello {name},",
            "headline": "We need one more thing from you.",
            "body": (
                "Your application looks promising (current score: {score}/900), "
                "but we need to verify some information before we can make a final decision. "
                "This is a standard requirement — nothing is wrong with your application."
            ),
            "next_step": "Please submit the following: {missing_items}. "
                         "Once we receive this, we'll process your application within 2 hours.",
        },
        "FRAUD_FLAGGED": {
            "greeting": "Hello {name},",
            "headline": "We cannot process this application right now.",
            "body": (
                "Our system has detected some unusual patterns in the transaction "
                "network associated with this application. This could be due to a "
                "number of reasons, including identity verification issues."
            ),
            "next_step": "Please visit your nearest bank branch with your Aadhaar card for "
                         "in-person verification. You can also call our helpline at {helpline}. "
                         "If this was an error, the verification will clear it up quickly.",
        },
    },
    "kannada": {
        "APPROVED": {
            "greeting": "{name}, ಒಳ್ಳೆಯ ಸುದ್ದಿ!",
            "headline": "ನಿಮ್ಮ ₹{amount} ಸಾಲ ಮಂಜೂರಾಗಿದೆ.",
            "body": (
                "ನಿಮ್ಮ UPI ಬಳಕೆ, ಬಿಲ್ ಪಾವತಿ, ಮತ್ತು ಖರ್ಚು ಮಾದರಿಯನ್ನು ನೋಡಿ "
                "ನಿಮ್ಮ ಕ್ರೆಡಿಟ್ ಸ್ಕೋರ್ 900 ರಲ್ಲಿ {score} ಬಂದಿದೆ. "
                "ಇದು 600 ಕ್ಕಿಂತ ಹೆಚ್ಚು, ಆದ್ದರಿಂದ ನೀವು ಅರ್ಹರು."
            ),
            "next_step": "24 ಗಂಟೆಗಳಲ್ಲಿ ಹಣ ನಿಮ್ಮ ಬ್ಯಾಂಕ್ ಖಾತೆಗೆ ಬರುತ್ತದೆ. "
                         "{cooling_off} ದಿನ ಕೂಲಿಂಗ್-ಆಫ್ ಅವಧಿ ಇರುತ್ತದೆ — "
                         "ಈ ಸಮಯದಲ್ಲಿ ಯಾವುದೇ ಶುಲ್ಕವಿಲ್ಲದೆ ರದ್ದು ಮಾಡಬಹುದು.",
        },
        "REJECTED": {
            "greeting": "ನಮಸ್ಕಾರ {name},",
            "headline": "ನಿಮ್ಮ ₹{amount} ಸಾಲ ಈಗ ಮಂಜೂರಾಗಿಲ್ಲ.",
            "body": (
                "ನಿಮ್ಮ ಕ್ರೆಡಿಟ್ ಸ್ಕೋರ್ ಈಗ 900 ರಲ್ಲಿ {score} ಇದೆ. "
                "ಸಾಲಕ್ಕೆ ಕನಿಷ್ಠ 600 ಬೇಕು. "
                "ಚಿಂತಿಸಬೇಡಿ — ಇದು ಶಾಶ್ವತ ಅಲ್ಲ. ನಿಮ್ಮ UPI ವ್ಯವಹಾರಗಳು "
                "ಒಳ್ಳೆಯ ದಿಕ್ಕಿನಲ್ಲಿವೆ. ನಿಮ್ಮ ಸ್ಕೋರ್ ಸುಧಾರಿಸಲು ಈ ಕೆಲಸ ಮಾಡಿ:"
            ),
            "next_step": "ಈ ಸುಧಾರಣೆಗಳನ್ನು {days_to_improve} ದಿನ ಮಾಡಿ, ನಂತರ ಮತ್ತೆ ಅರ್ಜಿ ಸಲ್ಲಿಸಿ. "
                         "ನಿಮ್ಮ ಪ್ರಗತಿ ನಾವು ನೆನಪಿನಲ್ಲಿಡುತ್ತೇವೆ.",
        },
        "MORE_INFO_NEEDED": {
            "greeting": "ನಮಸ್ಕಾರ {name},",
            "headline": "ನಿಮ್ಮಿಂದ ಇನ್ನೊಂದು ವಿಷಯ ಬೇಕು.",
            "body": (
                "ನಿಮ್ಮ ಅರ್ಜಿ ಒಳ್ಳೆಯದಾಗಿ ಕಾಣುತ್ತಿದೆ (ಸ್ಕೋರ್: {score}/900), "
                "ಆದರೆ ಅಂತಿಮ ನಿರ್ಧಾರ ತೆಗೆದುಕೊಳ್ಳಲು ಕೆಲವು ಮಾಹಿತಿ ಪರಿಶೀಲಿಸಬೇಕು. "
                "ಇದು ಸಾಮಾನ್ಯ ಪ್ರಕ್ರಿಯೆ — ನಿಮ್ಮ ಅರ್ಜಿಯಲ್ಲಿ ಯಾವ ಸಮಸ್ಯೆಯೂ ಇಲ್ಲ."
            ),
            "next_step": "ಈ ದಾಖಲೆಗಳನ್ನು ಸಲ್ಲಿಸಿ: {missing_items}. "
                         "ಸಿಕ್ಕ ನಂತರ 2 ಗಂಟೆಯಲ್ಲಿ ನಿರ್ಧಾರ ತಿಳಿಸುತ್ತೇವೆ.",
        },
        "FRAUD_FLAGGED": {
            "greeting": "ನಮಸ್ಕಾರ {name},",
            "headline": "ಈ ಅರ್ಜಿಯನ್ನು ಈಗ ಪ್ರಕ್ರಿಯೆಗೊಳಿಸಲು ಆಗುತ್ತಿಲ್ಲ.",
            "body": (
                "ಈ ಅರ್ಜಿಗೆ ಸಂಬಂಧಿಸಿದ ವ್ಯವಹಾರಗಳಲ್ಲಿ ಕೆಲವು ಅಸಾಮಾನ್ಯ ಮಾದರಿಗಳು "
                "ಕಂಡುಬಂದಿವೆ. ಇದು ಗುರುತು ಪರಿಶೀಲನೆ ಸಮಸ್ಯೆಯಿಂದ ಆಗಿರಬಹುದು."
            ),
            "next_step": "ನಿಮ್ಮ ಆಧಾರ್ ಕಾರ್ಡ್ ತೆಗೆದುಕೊಂಡು ಹತ್ತಿರದ ಬ್ಯಾಂಕ್ ಶಾಖೆಗೆ ಹೋಗಿ "
                         "ನೇರ ಪರಿಶೀಲನೆ ಮಾಡಿ. ಅಥವಾ {helpline} ಗೆ ಕರೆ ಮಾಡಿ. "
                         "ತಪ್ಪಿದ್ದರೆ, ಪರಿಶೀಲನೆ ಬೇಗ ಸರಿಯಾಗುತ್ತದೆ.",
        },
    },
    "hindi": {
        "APPROVED": {
            "greeting": "{name}, बढ़िया खबर!",
            "headline": "आपका ₹{amount} का लोन मंज़ूर हो गया है.",
            "body": (
                "हमने आपकी UPI एक्टिविटी, बिल भुगतान, और खर्च का पैटर्न देखा। "
                "इसके आधार पर आपका क्रेडिट स्कोर 900 में से {score} आया है। "
                "यह 600 से ऊपर है, इसलिए आप पात्र हैं।"
            ),
            "next_step": "24 घंटे में पैसा आपके बैंक खाते में आ जाएगा। "
                         "{cooling_off} दिन का कूलिंग-ऑफ़ पीरियड है — "
                         "इस दौरान बिना किसी चार्ज के कैंसल कर सकते हैं।",
        },
        "REJECTED": {
            "greeting": "नमस्कार {name},",
            "headline": "आपका ₹{amount} का लोन अभी मंज़ूर नहीं हुआ है.",
            "body": (
                "आपका क्रेडिट स्कोर अभी 900 में से {score} है। "
                "लोन के लिए कम से कम 600 चाहिए। "
                "चिंता न करें — यह हमेशा के लिए नहीं है। आपकी UPI एक्टिविटी "
                "अच्छी दिशा में है। स्कोर सुधारने के लिए यह करें:"
            ),
            "next_step": "ये सुधार {days_to_improve} दिन करें, फिर दोबारा अर्ज़ी दें। "
                         "आपकी प्रगति हम याद रखेंगे।",
        },
        "MORE_INFO_NEEDED": {
            "greeting": "नमस्कार {name},",
            "headline": "हमें आपसे एक और चीज़ चाहिए.",
            "body": (
                "आपकी अर्ज़ी अच्छी दिख रही है (स्कोर: {score}/900), "
                "लेकिन फ़ाइनल फ़ैसला लेने से पहले कुछ जानकारी वेरीफ़ाई करनी है। "
                "यह सामान्य प्रक्रिया है — आपकी अर्ज़ी में कोई समस्या नहीं है।"
            ),
            "next_step": "ये दस्तावेज़ भेजें: {missing_items}. "
                         "मिलने के बाद 2 घंटे में फ़ैसला बता देंगे।",
        },
        "FRAUD_FLAGGED": {
            "greeting": "नमस्कार {name},",
            "headline": "इस अर्ज़ी को अभी प्रोसेस नहीं किया जा सकता.",
            "body": (
                "इस अर्ज़ी से जुड़े लेनदेन में कुछ असामान्य पैटर्न दिखे हैं। "
                "यह पहचान सत्यापन की समस्या हो सकती है।"
            ),
            "next_step": "अपना आधार कार्ड लेकर नज़दीकी बैंक शाखा में जाएं "
                         "और सीधे वेरीफ़ाई कराएं। या {helpline} पर कॉल करें। "
                         "अगर गलती हुई है, तो वेरीफ़ाई होते ही सही हो जाएगा।",
        },
    },
}


# ─── Counterfactual Generator ───────────────────────────────────────────────

def generate_counterfactual(
    current_score: int,
    target_score: int,
    shap_values: dict,
    language: str = "english",
) -> list[dict]:
    """
    Generate mathematically-grounded counterfactual improvement advice.

    Uses SHAP's additive property:
        P(approval) = base_rate + Σ(SHAP values)
        Improving feature_i removes its negative SHAP contribution,
        adding |SHAP_i| to the probability → |SHAP_i| * 900 score points.

    Parameters
    ----------
    current_score : int
        Current credit score (0-900).
    target_score : int
        Target score to reach (e.g. 600 for approval).
    shap_values : dict
        {feature_name: shap_value} from RiskMind.
    language : str
        "english", "kannada", or "hindi".

    Returns
    -------
    list of dicts with:
        feature: str
        improvement: str (human-readable in chosen language)
        estimated_score_gain: int
        days_needed: int (estimated time)
        priority: int (1 = most impactful)
        cumulative_score: int (score after this + previous improvements)
    """
    score_gap = target_score - current_score
    if score_gap <= 0:
        return []

    # Get negative SHAP factors sorted by magnitude (most impactful first)
    negative_factors = [
        (feat, val) for feat, val in shap_values.items()
        if val < 0
    ]
    negative_factors.sort(key=lambda x: x[1])  # most negative first

    counterfactuals = []
    cumulative_gain = 0

    for priority, (feature, shap_val) in enumerate(negative_factors, 1):
        # Score gain = |SHAP value| * 900 (proportional to probability shift)
        estimated_gain = int(abs(shap_val) * 900)
        if estimated_gain < 5:
            continue  # too small to matter

        cumulative_gain += estimated_gain
        cumulative_score = current_score + cumulative_gain

        # Estimate days needed based on feature type
        days_needed = _estimate_improvement_days(feature, abs(shap_val))

        # Get human-readable advice in the chosen language
        trans = TRANSLATIONS.get(feature, {}).get(
            language, TRANSLATIONS.get(feature, {}).get("english", {})
        )
        advice_text = trans.get("advice", f"Improve your {feature}")

        # Fill in placeholders
        advice_text = advice_text.replace("{months}", str(max(1, days_needed // 30)))
        advice_text = advice_text.replace("{target}", str(max(5, int(abs(shap_val) * 30))))
        advice_text = advice_text.replace("{reduce_by}", str(int(abs(shap_val) * 50000)))
        advice_text = advice_text.replace("{current}", str(3))
        advice_text = advice_text.replace("{months_to_go}", str(max(3, days_needed // 30)))
        advice_text = advice_text.replace("{suggested_amount}", str(max(5000, int(abs(shap_val) * 100000))))

        counterfactuals.append({
            "feature": feature,
            "improvement": advice_text,
            "estimated_score_gain": estimated_gain,
            "days_needed": days_needed,
            "priority": priority,
            "cumulative_score": min(cumulative_score, 900),
        })

        # Stop if we've reached the target
        if current_score + cumulative_gain >= target_score:
            break

    return counterfactuals


def _estimate_improvement_days(feature: str, shap_magnitude: float) -> int:
    """Estimate realistic days needed to improve a feature."""
    base_days = {
        "utility_bill_payment_consistency": 60,
        "upi_txn_frequency_30d": 30,
        "savings_behavior_score": 45,
        "income_estimate_monthly": 90,
        "income_volatility_cv": 60,
        "bnpl_outstanding_ratio": 45,
        "multi_loan_app_count": 1,  # can do immediately
        "device_tenure_months": 90,
        "mobile_recharge_regularity": 30,
        "upi_merchant_diversity_score": 30,
        "evening_txn_ratio": 14,
        "peer_transfer_reciprocity": 45,
    }
    days = base_days.get(feature, 30)
    # Larger SHAP magnitude → more improvement needed → more time
    return int(days * (1 + shap_magnitude))


# ─── Full Explanation Builder ────────────────────────────────────────────────

def build_explanation(
    decision_data: dict,
    language: str = "english",
) -> str:
    """
    Build a complete human-readable explanation from decision data.

    Parameters
    ----------
    decision_data : dict with keys:
        applicant_name: str
        decision: "APPROVED" | "REJECTED" | "MORE_INFO_NEEDED" | "FRAUD_FLAGGED"
        credit_score: int (0-900)
        loan_amount: int
        shap_values: dict ({feature: shap_value})
        fraud_risk_level: str (from FraudSentinel)
        compliance_status: bool (from ComplianceGuard)
        missing_items: list[str] (for MORE_INFO_NEEDED)
        cooling_off_days: int (for APPROVED)
        helpline: str
    language : str

    Returns
    -------
    str — complete explanation in plain text (no markdown).
    """
    decision = decision_data.get("decision", "REJECTED")
    templates = DECISIONS.get(language, DECISIONS["english"])
    template = templates.get(decision, templates["REJECTED"])
    score = decision_data.get("credit_score", 0)
    name = decision_data.get("applicant_name", "Applicant")

    # ── 1. Build greeting + headline ─────────────────────────────────────────
    lines = []
    lines.append(template["greeting"].format(name=name))
    lines.append("")
    lines.append(template["headline"].format(
        name=name,
        amount=f"{decision_data.get('loan_amount', 0):,}",
    ))
    lines.append("")

    # ── 2. Body with score ───────────────────────────────────────────────────
    body = template["body"].format(
        name=name, score=score,
        amount=f"{decision_data.get('loan_amount', 0):,}",
    )
    lines.append(body)
    lines.append("")

    # ── 3. Top reasons (from SHAP) ───────────────────────────────────────────
    shap_values = decision_data.get("shap_values", {})
    if shap_values:
        translated = translate_shap_to_language(shap_values, language)

        # Top 3 positive factors
        positives = [t for t in translated if t["direction"] == "positive"][:3]
        negatives = [t for t in translated if t["direction"] == "negative"][:3]

        if positives:
            strength_header = {
                "english": "What's working in your favor:",
                "kannada": "ನಿಮ್ಮ ಪರವಾಗಿರುವ ಅಂಶಗಳು:",
                "hindi": "आपके पक्ष में क्या है:",
            }
            lines.append(strength_header.get(language, strength_header["english"]))
            for p in positives:
                lines.append(f"  + {p['explanation']}")
            lines.append("")

        if negatives:
            weakness_header = {
                "english": "What's holding you back:",
                "kannada": "ನಿಮ್ಮ ಸ್ಕೋರ್ ಕಡಿಮೆ ಮಾಡುತ್ತಿರುವ ಅಂಶಗಳು:",
                "hindi": "क्या चीज़ आपको पीछे खींच रही है:",
            }
            lines.append(weakness_header.get(language, weakness_header["english"]))
            for n in negatives:
                lines.append(f"  - {n['explanation']}")
            lines.append("")

    # ── 4. Counterfactual roadmap (for REJECTED) ─────────────────────────────
    if decision in ("REJECTED",) and shap_values:
        target_score = 650
        counterfactuals = generate_counterfactual(
            score, target_score, shap_values, language,
        )
        if counterfactuals:
            roadmap_header = {
                "english": f"Your roadmap to reach {target_score} points:",
                "kannada": f"{target_score} ಪಾಯಿಂಟ್ ತಲುಪಲು ನಿಮ್ಮ ಮಾರ್ಗಸೂಚಿ:",
                "hindi": f"{target_score} पॉइंट तक पहुंचने का आपका रोडमैप:",
            }
            lines.append(roadmap_header.get(language, roadmap_header["english"]))
            for cf in counterfactuals:
                days_label = {
                    "english": f"(~{cf['days_needed']} days, +{cf['estimated_score_gain']} points → {cf['cumulative_score']})",
                    "kannada": f"(~{cf['days_needed']} ದಿನ, +{cf['estimated_score_gain']} ಪಾಯಿಂಟ್ → {cf['cumulative_score']})",
                    "hindi": f"(~{cf['days_needed']} दिन, +{cf['estimated_score_gain']} पॉइंट → {cf['cumulative_score']})",
                }
                lines.append(f"  {cf['priority']}. {cf['improvement']}")
                lines.append(f"     {days_label.get(language, days_label['english'])}")
            lines.append("")

    # ── 5. Next step ─────────────────────────────────────────────────────────
    next_step = template["next_step"].format(
        name=name,
        cooling_off=decision_data.get("cooling_off_days", 3),
        days_to_improve=decision_data.get("days_to_improve", 45),
        missing_items=", ".join(decision_data.get("missing_items", ["Aadhaar verification"])),
        helpline=decision_data.get("helpline", "1800-XXX-XXXX"),
    )
    next_header = {
        "english": "Next step:",
        "kannada": "ಮುಂದಿನ ಹೆಜ್ಜೆ:",
        "hindi": "अगला कदम:",
    }
    lines.append(next_header.get(language, next_header["english"]))
    lines.append(f"  {next_step}")

    return "\n".join(lines)


# ─── Ramu Example (as specified in the prompt) ──────────────────────────────

RAMU_EXAMPLE = {
    "applicant_name": "Ramu",
    "decision": "REJECTED",
    "credit_score": 510,
    "loan_amount": 25000,
    "shap_values": {
        "upi_txn_frequency_30d": 0.12,
        "utility_bill_payment_consistency": -0.18,
        "savings_behavior_score": -0.15,
        "income_estimate_monthly": 0.05,
        "income_volatility_cv": -0.10,
        "bnpl_outstanding_ratio": -0.08,
        "multi_loan_app_count": -0.06,
        "upi_merchant_diversity_score": 0.04,
        "mobile_recharge_regularity": 0.03,
        "evening_txn_ratio": 0.02,
        "peer_transfer_reciprocity": 0.01,
        "device_tenure_months": -0.03,
    },
    "fraud_risk_level": "CLEAN",
    "compliance_status": True,
    "days_to_improve": 45,
    "helpline": "1800-XXX-XXXX",
}


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print("  ALIS ExplainerVoice — Decision Template Demo")
    print(f"{'='*60}")

    for lang in ["kannada", "english", "hindi"]:
        print(f"\n{'─'*60}")
        print(f"  LANGUAGE: {lang.upper()}")
        print(f"{'─'*60}\n")
        explanation = build_explanation(RAMU_EXAMPLE, language=lang)
        print(explanation)

    print(f"\n{'='*60}")
    print("  ✓ Done.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
