"""
ALIS — LoanOrchestrator
==============================================
The master controller for the ALIS multi-agent system.
Built using LangGraph to coordinate the 5 agents:
1. DataHarvester -> 2. FraudSentinel -> 3. RiskMind -> 4. ComplianceGuard -> 5. ExplainerVoice

Handles routing logic, error recovery, Thin-File Protocol, and Human Escalation.

INTEGRATION STRATEGY:
  - Each node FIRST tries to call the real agent code
  - If the agent's model artifacts don't exist (not trained yet),
    it falls back to simulation mode with a clear log message
  - This allows the system to work in BOTH demo mode and full mode

Usage:
    python orchestrator.py  # runs the full pipeline with test cases
"""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from langgraph.graph import StateGraph, START, END

# ─── Path Setup ──────────────────────────────────────────────────────────────

AGENTS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(AGENTS_DIR))
sys.path.insert(0, str(AGENTS_DIR / "data_harvester"))
sys.path.insert(0, str(AGENTS_DIR / "fraud_sentinel"))
sys.path.insert(0, str(AGENTS_DIR / "risk_mind"))
sys.path.insert(0, str(AGENTS_DIR / "compliance_guard"))
sys.path.insert(0, str(AGENTS_DIR / "explainer_voice"))

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LoanOrchestrator")


# ─── 1. State Schema ─────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """LangGraph State representing the loan application lifecycle."""
    applicant_id: str
    application_data: Dict[str, Any]

    harvested_data: Dict[str, Any]
    fraud_result: Dict[str, Any]
    credit_result: Dict[str, Any]
    compliance_result: Dict[str, Any]

    final_explanation: str
    pipeline_status: str
    error_log: List[str]
    escalate_to_human: bool
    escalation_reason: str
    timestamp_each_stage: Dict[str, str]
    is_thin_file: bool
    agent_modes: Dict[str, str]  # tracks real vs fallback for each agent


# ─── Human Escalation Handler ────────────────────────────────────────────────

def handle_escalation(state: AgentState) -> None:
    """
    Called when an application is escalated for human review.
    In production, this logs to PostgreSQL and sends a notification.
    """
    reason = state.get("escalation_reason", "Unknown internal rule")
    logger.critical(f"⚠️ ESCALATION TO HUMAN REVIEW: Applicant {state['applicant_id']}")
    logger.critical(f"   Reason: {reason}")
    logger.critical(f"   Audit Trail Logged to database -> [Pending human action]")


# ─── Agent Nodes ─────────────────────────────────────────────────────────────

def node_data_harvester(state: AgentState) -> Dict:
    """Agent 1: DataHarvester — Validates, normalizes, and enriches input data."""
    logger.info("-> DataHarvester: Validating application inputs...")
    data = state["application_data"]
    mode = "real"

    try:
        from data_harvester.harvester import DataHarvester
        harvester = DataHarvester()
        result = harvester.harvest(data)

        harvested = result["harvested_data"]
        is_thin = result["is_thin_file"]

        if result["validation_errors"]:
            logger.warning(f"   Validation errors: {result['validation_errors']}")

        logger.info(f"   ✓ DataHarvester completed (features: {len(result['feature_vector'])})")

    except Exception as e:
        logger.warning(f"   DataHarvester import failed: {e}. Using passthrough mode.")
        mode = "fallback"
        harvested = data.copy()
        history_days = data.get("upi_history_days", 100)
        is_thin = history_days < 30

    if is_thin:
        logger.warning(
            f"   [THIN-FILE PROTOCOL ACTIVATED] Applicant has only "
            f"{harvested.get('upi_history_days', 0)} days of history. "
            "Starting ALIS 30-day Gamified Observation Period."
        )

    return {
        "harvested_data": harvested,
        "is_thin_file": is_thin,
        "pipeline_status": "DATA_HARVESTED",
        "agent_modes": {**state.get("agent_modes", {}), "DataHarvester": mode},
        "timestamp_each_stage": {
            **state.get("timestamp_each_stage", {}),
            "DataHarvester": datetime.now().isoformat(),
        },
    }


def node_fraud_sentinel(state: AgentState) -> Dict:
    """Agent 2: FraudSentinel — Graph-based fraud detection via GraphSAGE."""
    logger.info("-> FraudSentinel: Checking UPI transaction network...")
    data = state["harvested_data"]
    mode = "real"

    try:
        from fraud_sentinel.scorer import FraudScorer
        scorer = FraudScorer()

        # Use applicant_id to look up in the graph, or score a synthetic node
        applicant_id = state.get("applicant_id", "L_0010")
        result = scorer.score_applicant(applicant_id)

        fraud_res = {
            "risk_score": result["fraud_risk_score"],
            "risk_level": result["risk_level"],
            "explanation": "; ".join(result.get("explanation", [])),
            "gnn_probability": result.get("gnn_probability"),
            "suspicious_connections": [
                acc["account_id"]
                for acc in result.get("connected_suspicious_accounts", [])
            ],
        }
        logger.info(
            f"   ✓ FraudSentinel scored: {fraud_res['risk_level']} "
            f"(score: {fraud_res['risk_score']})"
        )

    except Exception as e:
        logger.warning(f"   FraudSentinel model not available: {e}. Using simulation.")
        mode = "fallback"

        # Simulation fallback using injected flags
        is_fraudster = data.get("_simulated_fraud", False)
        if is_fraudster:
            fraud_res = {
                "risk_score": 98.5,
                "risk_level": "BLOCK",
                "explanation": "High in-degree from known mule accounts. Velocity exceeds organic limits.",
                "gnn_probability": 0.96,
                "suspicious_connections": ["MULE_99x", "MULE_12y"],
            }
        else:
            fraud_res = {
                "risk_score": 12.0,
                "risk_level": "CLEAN",
                "explanation": "Normal P2P and merchant network cluster.",
                "gnn_probability": 0.04,
                "suspicious_connections": [],
            }

    return {
        "fraud_result": fraud_res,
        "pipeline_status": "FRAUD_CHECKED",
        "agent_modes": {**state.get("agent_modes", {}), "FraudSentinel": mode},
        "timestamp_each_stage": {
            **state.get("timestamp_each_stage", {}),
            "FraudSentinel": datetime.now().isoformat(),
        },
    }


def node_risk_mind(state: AgentState) -> Dict:
    """Agent 3: RiskMind — Credit scoring via XGBoost + SHAP explainability."""
    logger.info("-> RiskMind: Calculating credit score using XGBoost/SHAP...")
    data = state["harvested_data"]
    mode = "real"

    try:
        from risk_mind.explainer import RiskMindExplainer
        explainer = RiskMindExplainer()

        # Build feature dict from harvested data
        feature_names = [
            "upi_txn_frequency_30d", "upi_avg_txn_amount",
            "upi_merchant_diversity_score", "utility_bill_payment_consistency",
            "mobile_recharge_regularity", "savings_behavior_score",
            "income_estimate_monthly", "income_volatility_cv",
            "bnpl_outstanding_ratio", "multi_loan_app_count",
            "peer_transfer_reciprocity", "evening_txn_ratio",
            "device_tenure_months",
        ]
        features = {
            feat: float(data.get(feat, 0))
            for feat in feature_names
        }

        result = explainer.explain_decision(features)

        credit_res = {
            "score": result["credit_score"],
            "probability": result.get("probability", 0.5),
            "approved": result.get("approved", result["credit_score"] >= 600),
            "shap_values": result.get("shap_values", {}),
            "top_positive_factors": result.get("top_positive_factors", []),
            "top_negative_factors": result.get("top_negative_factors", []),
            "counterfactual_advice": result.get("counterfactual_advice", []),
        }
        logger.info(
            f"   ✓ RiskMind scored: {credit_res['score']}/900 "
            f"(approved: {credit_res['approved']})"
        )

    except Exception as e:
        logger.warning(f"   RiskMind model not available: {e}. Using simulation.")
        mode = "fallback"

        # Simulation fallback
        score = data.get("_simulated_score", 720)
        credit_res = {
            "score": score,
            "probability": max(0.1, min(score / 900, 0.95)),
            "approved": score >= 600,
            "shap_values": {
                "utility_bill_payment_consistency": 0.15,
                "upi_txn_frequency_30d": 0.12,
                "savings_behavior_score": -0.08,
                "upi_merchant_diversity_score": 0.08,
                "income_estimate_monthly": 0.05,
                "income_volatility_cv": -0.10,
                "bnpl_outstanding_ratio": -0.06,
                "multi_loan_app_count": -0.04,
                "mobile_recharge_regularity": 0.03,
                "evening_txn_ratio": -0.02,
                "peer_transfer_reciprocity": 0.01,
                "device_tenure_months": -0.03,
            },
            "top_positive_factors": [
                {"feature": "utility_bill_payment_consistency", "impact": 0.15},
                {"feature": "upi_txn_frequency_30d", "impact": 0.12},
            ],
            "top_negative_factors": [
                {"feature": "income_volatility_cv", "impact": -0.10},
                {"feature": "savings_behavior_score", "impact": -0.08},
            ],
            "counterfactual_advice": [],
        }

    return {
        "credit_result": credit_res,
        "pipeline_status": "CREDIT_SCORED",
        "agent_modes": {**state.get("agent_modes", {}), "RiskMind": mode},
        "timestamp_each_stage": {
            **state.get("timestamp_each_stage", {}),
            "RiskMind": datetime.now().isoformat(),
        },
    }


def node_compliance_guard(state: AgentState) -> Dict:
    """Agent 4: ComplianceGuard — RBI regulatory compliance via rules + RAG."""
    logger.info("-> ComplianceGuard: Verifying loan terms against RBI guidelines...")
    data = state["harvested_data"]
    mode = "real"

    try:
        from compliance_guard.compliance_checker import (
            auto_adjust_terms,
            check_loan_compliance,
        )

        # Build loan offer from harvested data
        loan_offer = {
            "apr": data.get("apr", 24.0),
            "disbursal_account_type": data.get("disbursal_account_type", "own"),
            "kyc_completed": data.get("kyc_completed", True),
            "credit_limit_auto_increase": data.get("credit_limit_auto_increase", False),
            "cooling_off_days": data.get("cooling_off_days", 5),
            "recovery_contact_hours": data.get(
                "recovery_contact_hours", {"start": 9, "end": 18}
            ),
        }

        # First check
        result = check_loan_compliance(loan_offer, use_rag=False)

        # Auto-correction loop (max 2 retries) for non-critical violations
        retries = 0
        while (
            not result["is_compliant"]
            and not result["has_critical_violations"]
            and retries < 2
        ):
            logger.info(
                f"   Auto-adjusting loan terms (attempt {retries + 1}/2)..."
            )
            adjusted = auto_adjust_terms(loan_offer, result["violations"])
            adjustments = adjusted.pop("_adjustments_made", [])
            for adj in adjustments:
                logger.info(f"     → {adj}")
            loan_offer = adjusted
            result = check_loan_compliance(loan_offer, use_rag=False)
            retries += 1

        comp_res = {
            "is_compliant": result["is_compliant"],
            "has_critical_violations": result["has_critical_violations"],
            "violation_count": result["violation_count"],
            "violations": result["violations"],
            "recommended_corrections": result["recommended_corrections"],
            "compliance_report": result["compliance_report"],
            "offer_hash": result["offer_hash"],
            "adjusted_offer": loan_offer,
        }
        logger.info(
            f"   ✓ ComplianceGuard: {'COMPLIANT' if comp_res['is_compliant'] else 'NON-COMPLIANT'} "
            f"({comp_res['violation_count']} violations)"
        )

    except Exception as e:
        logger.warning(f"   ComplianceGuard import failed: {e}. Using simulation.")
        mode = "fallback"

        has_critical = data.get("_simulated_compliance_critical", False)
        if has_critical:
            comp_res = {
                "is_compliant": False,
                "has_critical_violations": True,
                "violation_count": 1,
                "violations": [
                    {
                        "severity": "CRITICAL",
                        "rule": "THIRD_PARTY_DISBURSAL",
                        "rbi_clause": "3.1",
                        "explanation": "Loan disbursal to third-party account prohibited.",
                    }
                ],
                "recommended_corrections": [],
                "compliance_report": "CRITICAL: Third-party disbursal detected.",
                "offer_hash": "SIMULATED",
                "adjusted_offer": {},
            }
        else:
            comp_res = {
                "is_compliant": True,
                "has_critical_violations": False,
                "violation_count": 0,
                "violations": [],
                "recommended_corrections": [],
                "compliance_report": "All RBI compliance checks passed.",
                "offer_hash": "SIMULATED",
                "adjusted_offer": {},
            }

    # Check if we need to escalate
    escalate = False
    escalation_reason = ""
    if comp_res.get("has_critical_violations"):
        escalate = True
        violations = comp_res.get("violations", [{}])
        rule = violations[0].get("rule", "UNKNOWN") if violations else "UNKNOWN"
        escalation_reason = f"ComplianceGuard found CRITICAL violation: {rule}"
        handle_escalation({**state, "escalation_reason": escalation_reason})

    return {
        "compliance_result": comp_res,
        "escalate_to_human": escalate,
        "escalation_reason": escalation_reason,
        "pipeline_status": "COMPLIANCE_CHECKED",
        "agent_modes": {**state.get("agent_modes", {}), "ComplianceGuard": mode},
        "timestamp_each_stage": {
            **state.get("timestamp_each_stage", {}),
            "ComplianceGuard": datetime.now().isoformat(),
        },
    }


def node_explainer_voice(state: AgentState) -> Dict:
    """Agent 5: ExplainerVoice — Vernacular explanations via templates + LLM."""
    logger.info("-> ExplainerVoice: Generating vernacular explanation...")
    mode = "real"

    try:
        from explainer_voice.llm_engine import ExplainerLLM

        engine = ExplainerLLM()

        # Package decision data for ExplainerVoice
        riskmind = state.get("credit_result", {})
        fraud = state.get("fraud_result", {})
        comp = state.get("compliance_result", {})

        explanation = engine.generate_explanation_for_orchestrator(
            riskmind_result={
                "credit_score": riskmind.get("score", 0),
                "shap_values": riskmind.get("shap_values", {}),
            },
            fraud_result={"risk_level": fraud.get("risk_level", "CLEAN")},
            compliance_result=comp,
            applicant_name=state["harvested_data"].get("applicant_name", "Applicant"),
            loan_amount=state["harvested_data"].get("loan_amount", 10000),
            language="kannada",  # Default to Kannada for competition demo
        )
        logger.info("   ✓ Vernacular explanation generated successfully!")

    except Exception as e:
        logger.warning(f"   ExplainerVoice import failed: {e}. Using template fallback.")
        mode = "fallback"

        # Graceful fallback: generate a structured explanation without imports
        fraud = state.get("fraud_result", {})
        riskmind = state.get("credit_result", {})
        name = state["harvested_data"].get("applicant_name", "Applicant")
        score = riskmind.get("score", 0)
        fraud_level = fraud.get("risk_level", "CLEAN")

        if fraud_level in ("BLOCK", "HIGH_RISK"):
            explanation = (
                f"ನಮಸ್ಕಾರ {name},\n\n"
                f"ಈ ಅರ್ಜಿಯನ್ನು ಈಗ ಪ್ರಕ್ರಿಯೆಗೊಳಿಸಲು ಆಗುತ್ತಿಲ್ಲ.\n"
                f"ನಿಮ್ಮ ವ್ಯವಹಾರ ಜಾಲದಲ್ಲಿ ಅಸಾಮಾನ್ಯ ಮಾದರಿಗಳು ಕಂಡುಬಂದಿವೆ.\n\n"
                f"ಮುಂದಿನ ಹೆಜ್ಜೆ:\n"
                f"  ನಿಮ್ಮ ಆಧಾರ್ ಕಾರ್ಡ್ ತೆಗೆದುಕೊಂಡು ಹತ್ತಿರದ ಬ್ಯಾಂಕ್ ಶಾಖೆಗೆ ಹೋಗಿ ಪರಿಶೀಲನೆ ಮಾಡಿ."
            )
        elif score >= 600:
            explanation = (
                f"{name}, ಒಳ್ಳೆಯ ಸುದ್ದಿ!\n\n"
                f"ನಿಮ್ಮ ₹{state['harvested_data'].get('loan_amount', 0):,} ಸಾಲ ಮಂಜೂರಾಗಿದೆ.\n"
                f"ನಿಮ್ಮ ಕ್ರೆಡಿಟ್ ಸ್ಕೋರ್ 900 ರಲ್ಲಿ {score} ಬಂದಿದೆ.\n\n"
                f"ಮುಂದಿನ ಹೆಜ್ಜೆ:\n"
                f"  24 ಗಂಟೆಗಳಲ್ಲಿ ಹಣ ನಿಮ್ಮ ಬ್ಯಾಂಕ್ ಖಾತೆಗೆ ಬರುತ್ತದೆ."
            )
        else:
            explanation = (
                f"ನಮಸ್ಕಾರ {name},\n\n"
                f"ನಿಮ್ಮ ₹{state['harvested_data'].get('loan_amount', 0):,} ಸಾಲ ಈಗ ಮಂಜೂರಾಗಿಲ್ಲ.\n"
                f"ನಿಮ್ಮ ಕ್ರೆಡಿಟ್ ಸ್ಕೋರ್ ಈಗ 900 ರಲ್ಲಿ {score} ಇದೆ.\n"
                f"ಸಾಲಕ್ಕೆ ಕನಿಷ್ಠ 600 ಬೇಕು.\n\n"
                f"ಮುಂದಿನ ಹೆಜ್ಜೆ:\n"
                f"  45 ದಿನ ನಿಮ್ಮ UPI ಬಳಕೆ ಮತ್ತು ಬಿಲ್ ಪಾವತಿ ಸುಧಾರಿಸಿ, ನಂತರ ಮತ್ತೆ ಅರ್ಜಿ ಸಲ್ಲಿಸಿ."
            )

    return {
        "final_explanation": explanation,
        "pipeline_status": "COMPLETED",
        "agent_modes": {**state.get("agent_modes", {}), "ExplainerVoice": mode},
        "timestamp_each_stage": {
            **state.get("timestamp_each_stage", {}),
            "ExplainerVoice": datetime.now().isoformat(),
        },
    }


# ─── Conditional Routing Logic ───────────────────────────────────────────────

def route_after_fraud(state: AgentState) -> str:
    """Routing: If FraudSentinel flags BLOCK -> skip to ExplainerVoice."""
    level = state["fraud_result"].get("risk_level", "CLEAN")
    if level in ["BLOCK", "HIGH_RISK"]:
        logger.warning(
            f"   [ROUTING] Fraud level is {level}. Short-circuiting directly to ExplainerVoice."
        )
        return "explainer_voice"
    return "risk_mind"


def route_after_compliance(state: AgentState) -> str:
    """Routing: If Compliance flags critical -> end. Else -> ExplainerVoice."""
    if state.get("escalate_to_human", False):
        logger.warning(
            "   [ROUTING] Critical compliance violation. Escaping pipeline to human review."
        )
        return END
    return "explainer_voice"


# ─── Build StateGraph ────────────────────────────────────────────────────────

def build_orchestrator():
    """Compiles the ALIS multi-agent LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("data_harvester", node_data_harvester)
    workflow.add_node("fraud_sentinel", node_fraud_sentinel)
    workflow.add_node("risk_mind", node_risk_mind)
    workflow.add_node("compliance_guard", node_compliance_guard)
    workflow.add_node("explainer_voice", node_explainer_voice)

    # Add Edges
    workflow.add_edge(START, "data_harvester")
    workflow.add_edge("data_harvester", "fraud_sentinel")

    # Conditional route: FraudSentinel -> (RiskMind) OR (ExplainerVoice)
    workflow.add_conditional_edges("fraud_sentinel", route_after_fraud)

    workflow.add_edge("risk_mind", "compliance_guard")

    # Conditional route: ComplianceGuard -> (ExplainerVoice) OR (End)
    workflow.add_conditional_edges("compliance_guard", route_after_compliance)

    workflow.add_edge("explainer_voice", END)

    return workflow.compile()


# ─── Full Working Example ────────────────────────────────────────────────────

def main():
    print(f"\n{'='*70}")
    print("  ALIS LoanOrchestrator — LangGraph Multi-Agent Pipeline")
    print(f"{'='*70}\n")

    orchestrator = build_orchestrator()

    # ── Test A: Ramu the auto driver (CLEAN -> APPROVED) ──
    print(f"{'─'*60}")
    print("  CASE A: Ramu (Auto Driver) — Clean & Approved")
    print(f"{'─'*60}")

    ramu_input = {
        "applicant_id": "APP_RAMU_123",
        "application_data": {
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
            # Simulation fallback values (used only if real agents unavailable)
            "_simulated_fraud": False,
            "_simulated_score": 720,
            "_simulated_compliance_critical": False,
        },
        "error_log": [],
        "agent_modes": {},
    }

    start_time = time.time()
    result_ramu = orchestrator.invoke(ramu_input)
    elapsed = time.time() - start_time

    print(f"\n[FINAL EXPLANATION FROM PIPELINE]:")
    print(result_ramu["final_explanation"])
    print(f"\n--- Pipeline Audit Trail ---")
    for stage, t in result_ramu.get("timestamp_each_stage", {}).items():
        agent_mode = result_ramu.get("agent_modes", {}).get(stage, "unknown")
        print(f"  ✓ {stage}: completed at {t[11:19]} [{agent_mode}]")
    print(f"  Total pipeline time: {elapsed:.2f}s\n")


    # ── Test B: Suspicious applicant (FRAUD_FLAGGED -> SHORT-CIRCUIT) ──
    print(f"\n{'─'*60}")
    print("  CASE B: Suspicious Profile — Mule Ring Detection")
    print(f"{'─'*60}")

    suspicious_input = {
        "applicant_id": "APP_SUSP_999",
        "application_data": {
            "applicant_name": "Suspicious User",
            "loan_amount": 80000,
            "upi_history_days": 60,
            "evening_txn_ratio": 0.65,
            "multi_loan_app_count": 7,
            "_simulated_fraud": True,
            "_simulated_score": 0,
            "_simulated_compliance_critical": False,
        },
        "error_log": [],
        "agent_modes": {},
    }

    start_time = time.time()
    result_susp = orchestrator.invoke(suspicious_input)
    elapsed = time.time() - start_time

    print(f"\n[FINAL EXPLANATION FROM PIPELINE]:")
    print(result_susp.get("final_explanation", "No explanation generated."))
    print(f"\n--- Pipeline Audit Trail ---")
    for stage, t in result_susp.get("timestamp_each_stage", {}).items():
        agent_mode = result_susp.get("agent_modes", {}).get(stage, "unknown")
        print(f"  ✓ {stage}: completed at {t[11:19]} [{agent_mode}]")

    skipped = "RiskMind" not in result_susp["timestamp_each_stage"]
    print(f"\n  ✓ FraudSentinel SHORT-CIRCUITED pipeline? {skipped}")
    print(f"  Total pipeline time: {elapsed:.2f}s\n")


    # ── Test C: Thin-File Protocol Activation ──
    print(f"\n{'─'*60}")
    print("  CASE C: Thin-File Profile (Gamified Observation Protocol)")
    print(f"{'─'*60}")

    thin_input = {
        "applicant_id": "APP_NEW_111",
        "application_data": {
            "applicant_name": "Lakshmi",
            "loan_amount": 5000,
            "upi_history_days": 12,
            "_simulated_fraud": False,
            "_simulated_score": 480,
            "_simulated_compliance_critical": False,
        },
        "error_log": [],
        "agent_modes": {},
    }

    start_time = time.time()
    result_thin = orchestrator.invoke(thin_input)
    elapsed = time.time() - start_time

    print(f"\n  ✓ Thin-File Protocol Triggered: {result_thin['is_thin_file']}")
    print(f"\n[FINAL EXPLANATION FROM PIPELINE]:")
    print(result_thin.get("final_explanation", "No explanation generated."))
    print(f"\n--- Pipeline Audit Trail ---")
    for stage, t in result_thin.get("timestamp_each_stage", {}).items():
        agent_mode = result_thin.get("agent_modes", {}).get(stage, "unknown")
        print(f"  ✓ {stage}: completed at {t[11:19]} [{agent_mode}]")
    print(f"  Total pipeline time: {elapsed:.2f}s")


    # ── Test D: Compliance Critical Failure ──
    print(f"\n\n{'─'*60}")
    print("  CASE D: Compliance Critical — Third-Party Disbursal Block")
    print(f"{'─'*60}")

    compliance_fail_input = {
        "applicant_id": "APP_COMP_777",
        "application_data": {
            "applicant_name": "Test Compliance",
            "loan_amount": 30000,
            "upi_history_days": 200,
            "disbursal_account_type": "third_party",
            "kyc_completed": False,
            "_simulated_fraud": False,
            "_simulated_score": 650,
            "_simulated_compliance_critical": True,
        },
        "error_log": [],
        "agent_modes": {},
    }

    start_time = time.time()
    result_comp = orchestrator.invoke(compliance_fail_input)
    elapsed = time.time() - start_time

    print(f"\n  ✓ Escalated to Human: {result_comp.get('escalate_to_human', False)}")
    print(f"  ✓ Reason: {result_comp.get('escalation_reason', 'N/A')}")
    print(f"  ✓ Pipeline Status: {result_comp.get('pipeline_status', 'N/A')}")
    print(f"  Note: ExplainerVoice was SKIPPED (compliance hard-block)")

    # Check if ExplainerVoice was actually skipped
    ev_skipped = "ExplainerVoice" not in result_comp.get("timestamp_each_stage", {})
    print(f"  ✓ ExplainerVoice skipped? {ev_skipped}")
    print(f"  Total pipeline time: {elapsed:.2f}s")

    # ── Summary ──
    print(f"\n\n{'='*70}")
    print("  PIPELINE SUMMARY")
    print(f"{'='*70}")
    print(f"  Case A (Ramu - Approved):    {result_ramu['pipeline_status']}")
    print(f"  Case B (Fraud - Blocked):    {result_susp['pipeline_status']}")
    print(f"  Case C (Thin File):          {result_thin['pipeline_status']}")
    print(f"  Case D (Compliance Block):   {result_comp['pipeline_status']}")

    # Show agent modes
    print(f"\n  Agent Integration Modes:")
    for agent, mode in result_ramu.get("agent_modes", {}).items():
        icon = "🟢" if mode == "real" else "🟡"
        print(f"    {icon} {agent}: {mode}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
