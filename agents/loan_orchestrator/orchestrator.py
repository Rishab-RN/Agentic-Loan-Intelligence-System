"""
ALIS — LoanOrchestrator
==============================================
The master controller for the ALIS multi-agent system.
Built using LangGraph to coordinate the 5 agents:
1. DataHarvester -> 2. FraudSentinel -> 3. RiskMind -> 4. ComplianceGuard -> 5. ExplainerVoice

Handles routing logic, error recovery, Thin-File Protocol, and Human Escalation.

Usage:
    python orchestrator.py  # runs the full pipeline with 2 test cases
"""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, List, TypedDict

from langgraph.graph import StateGraph, START, END

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


# ─── 4. Human Escalation Handler ─────────────────────────────────────────────

def handle_escalation(state: AgentState) -> None:
    """
    Called when an application is escalated for human review.
    In production, this logs to PostgreSQL and sends a notification.
    """
    reason = state.get("escalation_reason", "Unknown internal rule")
    logger.critical(f"⚠️ ESCALATION TO HUMAN REVIEW: Applicant {state['applicant_id']}")
    logger.critical(f"   Reason: {reason}")
    logger.critical(f"   Audit Trail Logged to database -> [Pending human action]")


# ─── 3. Agent Nodes ──────────────────────────────────────────────────────────

def node_data_harvester(state: AgentState) -> Dict:
    """Agent 1: DataHarvester (Collects and validates input data)."""
    logger.info("-> DataHarvester: Validating application inputs...")
    data = state["application_data"]
    
    # Simulating data harvested from UPI patterns and device logs
    harvested = data.copy()
    
    # Thin-File Protocol Check (less than 30 days history)
    history_days = data.get("upi_history_days", 100)
    is_thin = history_days < 30
    
    if is_thin:
        logger.warning(
            f"   [THIN-FILE PROTOCOL ACTIVATED] Applicant has only {history_days} days of history. "
            "Starting ALIS 30-day Gamified Observation Period. Score will be calibrated differently."
        )
    
    return {
        "harvested_data": harvested,
        "is_thin_file": is_thin,
        "pipeline_status": "DATA_HARVESTED",
        "timestamp_each_stage": {**state.get("timestamp_each_stage", {}), "DataHarvester": datetime.now().isoformat()}
    }


def node_fraud_sentinel(state: AgentState) -> Dict:
    """Agent 2: FraudSentinel (Graph-based fraud check)."""
    logger.info("-> FraudSentinel: Checking UPI transaction network...")
    data = state["harvested_data"]
    
    # In production, this would call `fraud_sentinel.scorer.score_fraud_risk`
    # For demo, we use the raw data flags injected from tests
    is_fraudster = data.get("_simulated_fraud", False)
    
    if is_fraudster:
        fraud_res = {
            "risk_score": 98.5,
            "risk_level": "BLOCK",
            "explanation": "High in-degree from known mule accounts suspected. Velocity exceeds organic limits.",
            "suspicious_connections": ["MULE_99x", "MULE_12y"]
        }
    else:
        fraud_res = {
            "risk_score": 12.0,
            "risk_level": "CLEAN",
            "explanation": "Normal P2P and merchant network cluster.",
            "suspicious_connections": []
        }
        
    return {
        "fraud_result": fraud_res,
        "pipeline_status": "FRAUD_CHECKED",
        "timestamp_each_stage": {**state.get("timestamp_each_stage", {}), "FraudSentinel": datetime.now().isoformat()}
    }


def node_risk_mind(state: AgentState) -> Dict:
    """Agent 3: RiskMind (Credit scoring via alternative data)."""
    logger.info("-> RiskMind: Calculating credit score using XGBoost/SHAP...")
    data = state["harvested_data"]
    
    # In production, this calls `risk_mind_model.predict_proba` and SHAP Explainer
    # Mocking appropriate scores for the demo logic
    score = data.get("_simulated_score", 720)
    
    credit_res = {
        "score": score,
        "approved": score >= 600,
        "shap_values": {
            "utility_bill_payment_consistency": 0.15,
            "upi_merchant_diversity_score": 0.08,
            "savings_behavior_score": -0.02
        }
        # In actual system, counterfactuals are generated in ExplainerVoice, but could happen here.
    }
    
    return {
        "credit_result": credit_res,
        "pipeline_status": "CREDIT_SCORED",
        "timestamp_each_stage": {**state.get("timestamp_each_stage", {}), "RiskMind": datetime.now().isoformat()}
    }


def node_compliance_guard(state: AgentState) -> Dict:
    """Agent 4: ComplianceGuard (RBI Rules check via RAG & Rules)."""
    logger.info("-> ComplianceGuard: Verifying loan terms against RBI digital lending guidelines...")
    data = state["harvested_data"]
    
    # In production, calls `compliance_checker.check_loan_compliance`
    # We will simulate a CRITICAL failure if the offer violates RBI guidelines deeply
    has_critical = data.get("_simulated_compliance_critical", False)
    
    if has_critical:
        comp_res = {
            "is_compliant": False,
            "violations": [{"severity": "CRITICAL", "rule": "THIRD_PARTY_DISBURSAL", "rbi_clause": "3.1"}],
            "recommended_corrections": []
        }
    else:
        comp_res = {
            "is_compliant": True,
            "violations": [],
            "recommended_corrections": []
        }
        
    # Check if we need to escalate
    escalate = False
    escalation_reason = ""
    if comp_res.get("violations") and any(v["severity"] == "CRITICAL" for v in comp_res["violations"]):
        escalate = True
        escalation_reason = f"ComplianceGuard found CRITICAL violation: {comp_res['violations'][0]['rule']}"
        handle_escalation({**state, "escalation_reason": escalation_reason})
        
    return {
        "compliance_result": comp_res,
        "escalate_to_human": escalate,
        "escalation_reason": escalation_reason,
        "pipeline_status": "COMPLIANCE_CHECKED",
        "timestamp_each_stage": {**state.get("timestamp_each_stage", {}), "ComplianceGuard": datetime.now().isoformat()}
    }


def node_explainer_voice(state: AgentState) -> Dict:
    """Agent 5: ExplainerVoice (Vernacular explanations via LLM/Templates)."""
    logger.info("-> ExplainerVoice: Generating vernacular explanation for user...")
    
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    try:
        from explainer_voice.llm_engine import ExplainerLLM
        engine = ExplainerLLM()
        
        # Package decision for ExplainerVoice
        riskmind = state.get("credit_result", {})
        fraud = state.get("fraud_result", {})
        comp = state.get("compliance_result", {})
        
        explanation = engine.generate_explanation_for_orchestrator(
            riskmind_result={"credit_score": riskmind.get("score", 0), "shap_values": riskmind.get("shap_values", {})},
            fraud_result={"risk_level": fraud.get("risk_level", "CLEAN")},
            compliance_result=comp,
            applicant_name=state["harvested_data"].get("applicant_name", "Applicant"),
            loan_amount=state["harvested_data"].get("loan_amount", 10000),
            language="kannada"  # Defaulting to Kannada for competition demo
        )
    except Exception as e:
        logger.warning(f"Failed to load real ExplainerVoice: {e}. Falling back to mock.")
        explanation = "[MOCK EXPLANATION] ನಮಸ್ಕಾರ, ನಿಮ್ಮ ಸಾಲದ ಡೇಟಾ ಪರಿಶೀಲಿಸಲಾಗಿದೆ..."

    logger.info("   [Vernacular explanation generated successfully!]")
    
    return {
        "final_explanation": explanation,
        "pipeline_status": "COMPLETED",
        "timestamp_each_stage": {**state.get("timestamp_each_stage", {}), "ExplainerVoice": datetime.now().isoformat()}
    }


# ─── 2. Conditional Routing Logic ────────────────────────────────────────────

def route_after_fraud(state: AgentState) -> str:
    """Routing: If FraudSentinel flags BLOCK -> skip to ExplainerVoice."""
    level = state["fraud_result"].get("risk_level", "CLEAN")
    if level in ["BLOCK", "HIGH_RISK"]:
        logger.warning(f"   [ROUTING] Fraud level is {level}. Short-circuiting directly to ExplainerVoice.")
        return "explainer_voice"
    return "risk_mind"


def route_after_compliance(state: AgentState) -> str:
    """Routing: If Compliance flags critical -> end. Else -> ExplainerVoice."""
    if state.get("escalate_to_human", False):
        logger.warning("   [ROUTING] Critical compliance violation. Escaping pipeline to human review.")
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


# ─── 5. Full Working Example ─────────────────────────────────────────────────

def main():
    print(f"\n{'='*70}")
    print("  ALIS LoanOrchestrator — LangGraph Multi-Agent Pipeline Demo")
    print(f"{'='*70}\n")
    
    orchestrator = build_orchestrator()

    # ── Test A: Ramu the auto driver (CLEAN -> APPROVED) ──
    print(f"{'─'*50}")
    print("  CASE A: Ramu (Auto Driver) — Clean & Approved")
    print(f"{'─'*50}")
    
    ramu_input = {
        "applicant_id": "APP_RAMU_123",
        "application_data": {
            "applicant_name": "Ramu",
            "loan_amount": 20000,
            "upi_history_days": 350,
            "_simulated_fraud": False,
            "_simulated_score": 750,
            "_simulated_compliance_critical": False
        },
        "error_log": []
    }
    
    result_ramu = orchestrator.invoke(ramu_input)
    print("\n[FINAL EXPLANATION FROM PIPELINE]:")
    print(result_ramu["final_explanation"])
    print("\n--- Pipeline Audit Trail ---")
    for stage, t in result_ramu.get("timestamp_each_stage", {}).items():
        print(f" * {stage}: completed at {t[11:19]}")


    # ── Test B: Suspicious applicant (FRAUD_FLAGGED -> SHORT-CIRCUIT) ──
    print(f"\n\n{'─'*50}")
    print("  CASE B: Suspicious Profile — Mule Ring Detection")
    print(f"{'─'*50}")

    suspicious_input = {
        "applicant_id": "APP_SUSP_999",
        "application_data": {
            "applicant_name": "Suspicious User",
            "loan_amount": 50000,
            "upi_history_days": 60,
            "_simulated_fraud": True,
            "_simulated_score": 0, # Unused because it short-circuits
            "_simulated_compliance_critical": False
        },
        "error_log": []
    }
    
    result_susp = orchestrator.invoke(suspicious_input)
    print("\n[FINAL EXPLANATION FROM PIPELINE]:")
    print(result_susp.get("final_explanation", "No explanation generated."))
    print("\n--- Pipeline Audit Trail ---")
    for stage, t in result_susp.get("timestamp_each_stage", {}).items():
        print(f" * {stage}: completed at {t[11:19]}")
        
    # Note: RiskMind and ComplianceGuard were SKIPPED in Case B!
    skipped = "RiskMind" not in result_susp["timestamp_each_stage"]
    print(f"\n✓ FraudSentinel SHORT-CIRCUITED pipeline? {skipped}")


    # ── Test C: Thin-File Protocol Activation ──
    print(f"\n\n{'─'*50}")
    print("  CASE C: Thin-File Profile (Gamified Observation Protocol)")
    print(f"{'─'*50}")

    thin_input = {
        "applicant_id": "APP_NEW_111",
        "application_data": {
            "applicant_name": "New User",
            "loan_amount": 5000,
            "upi_history_days": 12,  # < 30 triggers Thin File
            "_simulated_fraud": False,
            "_simulated_score": 620,
            "_simulated_compliance_critical": False
        },
        "error_log": []
    }
    
    result_thin = orchestrator.invoke(thin_input)
    print(f"\n✓ Thin-File Protocol Triggered: {result_thin['is_thin_file']}")


if __name__ == "__main__":
    main()
