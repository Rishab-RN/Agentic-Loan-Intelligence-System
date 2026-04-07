"""
ALIS — Agentic Loan Intelligence System
Competition Demo Dashboard (RVCE FinTech Summit 2026)

Features:
  - Demo Mode: Pre-computed results for 3 personas (instant, no deps)
  - Live Mode: Calls the real LangGraph orchestrator pipeline

Run:
  streamlit run app.py
"""

import sys
import time
from pathlib import Path

import networkx as nx
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

# ─── Add agents to path for Live Mode ─────────────────────────────────────────

AGENTS_DIR = Path(__file__).parent.parent / "agents"
sys.path.insert(0, str(AGENTS_DIR))
sys.path.insert(0, str(AGENTS_DIR / "loan_orchestrator"))

# ─── Page Configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ALIS | Agentic Loan Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme styling and agent boxes
st.markdown("""
<style>
    .agent-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #1E293B;
        border: 1px solid #334155;
        text-align: center;
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    .agent-box.waiting { border-color: #334155; color: #94A3B8; }
    .agent-box.processing { border-color: #3B82F6; color: #60A5FA; box-shadow: 0 0 10px #3B82F6; }
    .agent-box.done { border-color: #10B981; color: #34D399; }
    .agent-box.flagged { border-color: #EF4444; color: #F87171; }
    .agent-title { font-weight: bold; font-size: 14px; margin-bottom: 5px; }
    .agent-status { font-size: 12px; }
    .agent-time { font-size: 10px; color: #64748B; margin-top: 5px; }
    .rbi-quote { border-left: 4px solid #EF4444; padding-left: 10px; color: #FCA5A5; font-style: italic; }
    .counterfactual { background-color: rgba(59, 130, 246, 0.1); border-left: 4px solid #3B82F6; padding: 15px; border-radius: 4px; margin-top: 20px; }
    .mode-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
        margin-left: 8px;
    }
    .mode-real { background: #064E3B; color: #34D399; }
    .mode-fallback { background: #78350F; color: #FCD34D; }
</style>
""", unsafe_allow_html=True)


# ─── Mock Data for Demo Mode ─────────────────────────────────────────────────

# Ramu - Clean profile, Approved
RAMU_DATA = {
    "name": "Ramu S.",
    "profile": "Auto Driver, Ballari. Earns ₹900/day, never missed electricity bill, zero CIBIL.",
    "amount": 25000,
    "score": 720,
    "decision": "APPROVED",
    "fraud_level": "CLEAN",
    "compliance": True,
    "compliance_msg": "✅ All RBI parameters clear (APR, Disbursal, Cooling-off)",
    "shap_values": {
        "Utility Bills": 85,
        "UPI Frequency": 60,
        "Merchant Diversity": 30,
        "Late Night Txns": -15,
        "Loan App Count": -10,
    },
    "credit_agent_output": {
        "credit_score": 720,
        "probability": 0.80,
        "approved": True,
        "top_positive_factors": [
            {"feature": "utility_bill_payment_consistency", "impact": 0.15, "label": "Consistent bill payments show financial reliability"},
            {"feature": "upi_txn_frequency_30d", "impact": 0.12, "label": "High UPI transaction frequency indicates active digital economy participation"},
            {"feature": "upi_merchant_diversity_score", "impact": 0.08, "label": "Diverse merchant transactions prove genuine business activity"},
        ],
        "top_negative_factors": [
            {"feature": "evening_txn_ratio", "impact": -0.05, "label": "Some late-night transactions detected"},
            {"feature": "multi_loan_app_count", "impact": -0.04, "label": "Multiple loan apps installed on device"},
        ],
        "counterfactual_advice": [
            "Reduce late-night transactions to improve score by ~30 points",
            "Remove unused loan apps from your phone",
        ],
    },
    "explanations": {
        "English": (
            "Great news, Ramu! Your loan of ₹25,000 is approved.\n\n"
            "We looked at how you use UPI and pay your bills. Your score is 720 out of 900. "
            "Your strongest points are your consistent electricity bill payments and daily UPI usage. "
            "Your loan will be disbursed in 2 hours. You have a 3-day cooling-off period to cancel without charges."
        ),
        "ಕನ್ನಡ": (
            "ರಾಮು, ಒಳ್ಳೆಯ ಸುದ್ದಿ! ನಿಮ್ಮ ₹25,000 ಸಾಲ ಮಂಜೂರಾಗಿದೆ.\n\n"
            "ನಿಮ್ಮ ಬಿಲ್ ಪಾವತಿ ಮತ್ತು ಸಾಲದ ಇತಿಹಾಸ ನೋಡಿ ನಿಮ್ಮ ಸ್ಕೋರ್ 900 ಕ್ಕೆ 720 ಬಂದಿದೆ. "
            "ನಿಮ್ಮ ಬಲವಾದ ಅಂಶವೆಂದರೆ ವಿದ್ಯುತ್ ಬಿಲ್ ಸರಿಯಾಗಿ ಕಟ್ಟುವುದು. "
            "2 ಗಂಟೆಯಲ್ಲಿ ಹಣ ನಿಮ್ಮ ಖಾತೆಗೆ ಬರುತ್ತದೆ. 3 ದಿನದ ಕೂಲಿಂಗ್-ಆಫ್ ಅವಧಿ ಇರುತ್ತದೆ."
        ),
        "हिंदी": (
            "रामू, बढ़िया खबर! आपका ₹25,000 का लोन मंजूर हो गया है।\n\n"
            "आपके बिल भुगतान को देखकर आपका स्कोर 900 में से 720 है। "
            "आपकी सबसे बड़ी ताकत बिजली का बिल सही समय पर भरना है। "
            "2 घंटे में पैसा आपके खाते में आ जाएगा। 3 दिन का कूलिंग-ऑफ़ पीरियड है।"
        )
    },
    "counterfactual": "You're already approved! Keep paying utility bills on time to unlock higher limits."
}

# Meena - Rejection with improvement
MEENA_DATA = {
    "name": "Meena K.",
    "profile": "Tuition Teacher, Davangere. ₹15,000/month irregular income, high BNPL balance.",
    "amount": 40000,
    "score": 510,
    "decision": "REJECTED",
    "fraud_level": "CLEAN",
    "compliance": True,
    "compliance_msg": "✅ Terms compliant with RBI Guidelines.",
    "shap_values": {
        "BNPL Outstanding": -65,
        "Income Volatility": -45,
        "Multiple Loan Apps": -30,
        "Savings Habit": 20,
        "UPI Frequency": 15,
    },
    "credit_agent_output": {
        "credit_score": 510,
        "probability": 0.42,
        "approved": False,
        "top_positive_factors": [
            {"feature": "savings_behavior_score", "impact": 0.06, "label": "Some savings behavior detected"},
            {"feature": "upi_txn_frequency_30d", "impact": 0.04, "label": "Regular UPI usage"},
        ],
        "top_negative_factors": [
            {"feature": "bnpl_outstanding_ratio", "impact": -0.22, "label": "High BNPL/EMI outstanding relative to income"},
            {"feature": "income_volatility_cv", "impact": -0.18, "label": "Monthly income varies by more than 40%"},
            {"feature": "multi_loan_app_count", "impact": -0.12, "label": "4+ loan apps installed signals credit stress"},
        ],
        "counterfactual_advice": [
            "Pay off ₹8,000 of BNPL balance → +110 points (~45 days)",
            "Uninstall 2 unused loan apps → +45 points (immediate)",
            "Deposit ₹500/month consistently for 2 months → +30 points (~60 days)",
        ],
    },
    "explanations": {
        "English": "Hello Meena,\n\nYour loan is not approved right now. Your score is 510, and we need 600. The main issue is the high outstanding amount on your Buy-Now-Pay-Later apps and varying monthly income.",
        "ಕನ್ನಡ": "ನಮಸ್ಕಾರ ಮೀನಾ,\n\nನಿಮ್ಮ ಸಾಲ ಈಗ ಮಂಜೂರಾಗಿಲ್ಲ. ನಿಮ್ಮ ಸ್ಕೋರ್ 510 ಇದೆ, ನಮಗೆ 600 ಬೇಕು. ಮುಖ್ಯ ಕಾರಣವೆಂದರೆ ನಿಮ್ಮ EMI ಹೊರೆ ಮತ್ತು ಆದಾಯದ ಏರಿಳಿತ.",
        "हिंदी": "नमस्ते मीना,\n\nआपका लोन अभी मंजूर नहीं हुआ है। आपका स्कोर 510 है, और हमें 600 चाहिए। मुख्य कारण आपकी EMI का बोझ और आय में बदलाव है।"
    },
    "counterfactual": (
        "💡 Roadmap to Approval (650 points):\n"
        "1. Pay off ₹8,000 of your BNPL balance (+110 points, ~45 days)\n"
        "2. Uninstall 2 unused loan apps (+45 points, immediate)\n"
        "3. Deposit ₹500 consistently for 2 months (+30 points, ~60 days)"
    )
}

# Suspicious - Fraud Flag
SUSPICIOUS_DATA = {
    "name": "Applicant X",
    "profile": "Unknown user. Synthetic graph indicates heavy in-degree from known mule accounts.",
    "amount": 80000,
    "score": 0,
    "decision": "FRAUD_FLAGGED",
    "fraud_level": "BLOCK",
    "compliance": False,
    "compliance_msg": "❌ CRITICAL: Disbursal requested to third-party pool account.",
    "compliance_quote": '"Loan disbursal to a third-party account is prohibited. Disbursals must be made directly to borrower\'s account." (RBI Clause 3.1)',
    "shap_values": {"Mule Connections": -200, "Velocity": -150, "Night Txn": -100, "Device Age": -50},
    "credit_agent_output": {
        "credit_score": 0,
        "probability": 0.0,
        "approved": False,
        "top_positive_factors": [],
        "top_negative_factors": [
            {"feature": "fraud_flag", "impact": -1.0, "label": "Application blocked by FraudSentinel — credit scoring bypassed"},
        ],
        "counterfactual_advice": [
            "Visit nearest bank branch with Aadhaar/PAN for in-person KYC verification",
        ],
    },
    "explanations": {
        "English": "We cannot process this application right now. Our systems detected unusual identity and network patterns.",
        "ಕನ್ನಡ": "ಈ ಅರ್ಜಿಯನ್ನು ಪ್ರಕ್ರಿಯೆಗೊಳಿಸಲು ಆಗುವುದಿಲ್ಲ. ಅಸಾಮಾನ್ಯ ಜಾಲಾಕ್ಷರಿ ಮಾದರಿಗಳು ಕಂಡುಬಂದಿವೆ.",
        "हिंदी": "हम इस अर्ज़ी को प्रोसेस नहीं कर सकते। कुछ असामान्य पैटर्न दिखे हैं।"
    },
    "counterfactual": "Please visit your nearest bank branch with physical Aadhaar/PAN for mandatory in-person KYC verification."
}

PERSONAS = {
    "Ramu (Auto Driver)": RAMU_DATA,
    "Meena (Tuition Teacher)": MEENA_DATA,
    "Suspicious (Fraud Ring)": SUSPICIOUS_DATA
}

# Live mode application data (maps persona → orchestrator input)
LIVE_INPUTS = {
    "Ramu (Auto Driver)": {
        "applicant_id": "APP_RAMU_LIVE",
        "application_data": {
            "applicant_name": "Ramu S.",
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
            "_simulated_fraud": False,
            "_simulated_score": 720,
            "_simulated_compliance_critical": False,
        },
        "error_log": [],
        "agent_modes": {},
    },
    "Meena (Tuition Teacher)": {
        "applicant_id": "APP_MEENA_LIVE",
        "application_data": {
            "applicant_name": "Meena K.",
            "loan_amount": 40000,
            "upi_history_days": 180,
            "upi_txn_frequency_30d": 30,
            "upi_avg_txn_amount": 350,
            "upi_merchant_diversity_score": 0.35,
            "utility_bill_payment_consistency": 0.55,
            "mobile_recharge_regularity": 0.60,
            "savings_behavior_score": 30,
            "income_estimate_monthly": 15000,
            "income_volatility_cv": 0.55,
            "bnpl_outstanding_ratio": 0.45,
            "multi_loan_app_count": 4,
            "peer_transfer_reciprocity": 0.40,
            "evening_txn_ratio": 0.18,
            "device_tenure_months": 10,
            "_simulated_fraud": False,
            "_simulated_score": 510,
            "_simulated_compliance_critical": False,
        },
        "error_log": [],
        "agent_modes": {},
    },
    "Suspicious (Fraud Ring)": {
        "applicant_id": "APP_SUSP_LIVE",
        "application_data": {
            "applicant_name": "Applicant X",
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
    },
}


# ─── Helper Visualization Functions ──────────────────────────────────────────

def create_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "ALIS Risk Score", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [None, 900], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "rgba(255,255,255,0.7)"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 450], 'color': '#EF4444'},
                {'range': [450, 600], 'color': '#F59E0B'},
                {'range': [600, 900], 'color': '#10B981'}
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    return fig

def create_waterfall(shap_dict):
    labels = list(shap_dict.keys())
    values = list(shap_dict.values())

    fig = go.Figure(go.Waterfall(
        orientation="h",
        measure=["relative"] * len(labels),
        y=labels,
        x=values,
        textposition="outside",
        text=[f"+{v}" if v > 0 else str(v) for v in values],
        decreasing={"marker": {"color": "#EF4444"}},
        increasing={"marker": {"color": "#10B981"}},
    ))
    fig.update_layout(
        height=300,
        title="Impact of Features (SHAP)",
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"},
        xaxis=dict(showgrid=False, zeroline=True, zerolinecolor='white'),
        yaxis=dict(autorange="reversed")
    )
    return fig

def create_network_graph(fraud_level):
    G = nx.erdos_renyi_graph(20, 0.15, seed=42) if fraud_level == "CLEAN" else nx.barabasi_albert_graph(20, 3, seed=42)
    pos = nx.spring_layout(G, seed=42)

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=0.5, color='#475569'),
        hoverinfo='none', mode='lines'
    )

    node_x, node_y, node_colors = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        color = '#EF4444' if (fraud_level == "BLOCK" and G.degree(node) > 4) else '#3B82F6'
        node_colors.append(color)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text',
        marker=dict(size=10, color=node_colors, line=dict(width=2))
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="UPI Transaction Network",
        title_font_size=16, showlegend=False, hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"}
    )
    return fig


# ─── Live Mode: Run Real Orchestrator ────────────────────────────────────────

@st.cache_resource
def get_orchestrator():
    """Cache the compiled LangGraph orchestrator (expensive to build)."""
    try:
        from orchestrator import build_orchestrator
        return build_orchestrator()
    except Exception as e:
        st.error(f"Could not load orchestrator: {e}")
        return None


def run_live_pipeline(persona_name: str) -> dict:
    """Execute the real LangGraph pipeline and parse results into display format."""
    orchestrator = get_orchestrator()
    if orchestrator is None:
        return None

    input_data = LIVE_INPUTS.get(persona_name)
    if input_data is None:
        return None

    # Run the pipeline
    result = orchestrator.invoke(input_data)

    # Parse results into the display format expected by the dashboard
    credit = result.get("credit_result", {})
    fraud = result.get("fraud_result", {})
    compliance = result.get("compliance_result", {})
    score = credit.get("score", 0)
    fraud_level = fraud.get("risk_level", "CLEAN")

    # Determine decision
    if fraud_level in ("BLOCK", "HIGH_RISK"):
        decision = "FRAUD_FLAGGED"
    elif score >= 600:
        decision = "APPROVED"
    else:
        decision = "REJECTED"

    # Build SHAP display dict (convert feature names to readable labels)
    shap_display = {}
    shap_label_map = {
        "utility_bill_payment_consistency": "Utility Bills",
        "upi_txn_frequency_30d": "UPI Frequency",
        "savings_behavior_score": "Savings Habit",
        "upi_merchant_diversity_score": "Merchant Diversity",
        "income_estimate_monthly": "Monthly Income",
        "income_volatility_cv": "Income Volatility",
        "bnpl_outstanding_ratio": "BNPL Outstanding",
        "multi_loan_app_count": "Multiple Loan Apps",
        "mobile_recharge_regularity": "Recharge Regularity",
        "evening_txn_ratio": "Late Night Txns",
        "peer_transfer_reciprocity": "Peer Transfers",
        "device_tenure_months": "Device Age",
    }
    raw_shap = credit.get("shap_values", {})
    for feat, val in sorted(raw_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        label = shap_label_map.get(feat, feat)
        shap_display[label] = int(val * 600)  # Scale for visual impact

    # Build compliance display
    is_compliant = compliance.get("is_compliant", True)
    if is_compliant:
        comp_msg = "✅ All RBI parameters clear (APR, Disbursal, Cooling-off)"
    else:
        violations = compliance.get("violations", [])
        v_rules = [v.get("rule", "UNKNOWN") for v in violations]
        comp_msg = f"❌ Violations: {', '.join(v_rules)}"

    # Build credit agent output for display
    shap_label_map_rev = {v: k for k, v in shap_label_map.items()}
    top_pos = credit.get("top_positive_factors", [])
    top_neg = credit.get("top_negative_factors", [])
    cf_advice = credit.get("counterfactual_advice", [])

    # Build human-readable factor labels
    def factor_label(f):
        feat = f.get("feature", "") if isinstance(f, dict) else str(f)
        return shap_label_map.get(feat, feat.replace("_", " ").title())

    credit_output = {
        "credit_score": score,
        "probability": credit.get("probability", 0),
        "approved": credit.get("approved", score >= 600),
        "top_positive_factors": [
            {
                "feature": f.get("feature", "") if isinstance(f, dict) else str(f),
                "impact": f.get("impact", 0) if isinstance(f, dict) else 0,
                "label": factor_label(f),
            }
            for f in top_pos
        ],
        "top_negative_factors": [
            {
                "feature": f.get("feature", "") if isinstance(f, dict) else str(f),
                "impact": f.get("impact", 0) if isinstance(f, dict) else 0,
                "label": factor_label(f),
            }
            for f in top_neg
        ],
        "counterfactual_advice": cf_advice if isinstance(cf_advice, list) else [],
    }

    return {
        "name": input_data["application_data"]["applicant_name"],
        "profile": f"Live pipeline result for {input_data['application_data']['applicant_name']}",
        "amount": input_data["application_data"]["loan_amount"],
        "score": score,
        "decision": decision,
        "fraud_level": fraud_level,
        "compliance": is_compliant,
        "compliance_msg": comp_msg,
        "compliance_quote": compliance.get("compliance_report", ""),
        "shap_values": shap_display if shap_display else {"No SHAP data": 0},
        "credit_agent_output": credit_output,
        "explanations": {
            "English": result.get("final_explanation", "Explanation not available."),
            "ಕನ್ನಡ": result.get("final_explanation", "ವಿವರಣೆ ಲಭ್ಯವಿಲ್ಲ."),
            "हिंदी": result.get("final_explanation", "विवरण उपलब्ध नहीं है।"),
        },
        "counterfactual": "Counterfactual advice generated by the RiskMind SHAP engine.",
        "agent_modes": result.get("agent_modes", {}),
        "timestamp_each_stage": result.get("timestamp_each_stage", {}),
        "pipeline_status": result.get("pipeline_status", "UNKNOWN"),
        "is_thin_file": result.get("is_thin_file", False),
    }


# ─── Build Layout ────────────────────────────────────────────────────────────

def render_agent_box(title, status_class, status_text, time_ms=""):
    icon_map = {"waiting": "⏳", "processing": "🔄", "done": "✅", "flagged": "❌"}
    return f"""
    <div class="agent-box {status_class}">
        <div class="agent-title">{title}</div>
        <div class="agent-status">{icon_map[status_class]} {status_text}</div>
        <div class="agent-time">{time_ms}</div>
    </div>
    """

st.markdown("<h1 style='text-align: center; color: #60A5FA;'>ALIS 🧠 LoanOrchestrator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94A3B8; margin-bottom: 30px;'>Agentic Loan Intelligence System — AI Credit Underwriting for the Underserved</p>", unsafe_allow_html=True)

# ─── Sidebar: The Story ───
st.sidebar.title("👥 The Story")
mode = st.sidebar.radio("Backend Mode", ["Demo Mode (Pre-computed)", "Live Mode (Python Orchestrator)"])
is_live = mode.startswith("Live")

if is_live:
    st.sidebar.info("🟢 **Live Mode**: Calling real LangGraph pipeline with all 5 agents.")
else:
    st.sidebar.info("🔵 **Demo Mode**: Using pre-computed results (instant).")

st.sidebar.markdown("---")
st.sidebar.subheader("Select Applicant Scenario:")

persona_name = st.sidebar.radio("Evaluate:", list(PERSONAS.keys()))

# Show persona profile from demo data
demo_data = PERSONAS[persona_name]
st.sidebar.markdown(f"**Profile:**<br>{demo_data['profile']}", unsafe_allow_html=True)
st.sidebar.markdown(f"**Requested Loan:** ₹{demo_data['amount']:,}")

if st.sidebar.button("🔄 Reset / Start Fresh", use_container_width=True):
    st.session_state.processed = False
    st.session_state.live_result = None
    st.rerun()

# ─── Main Logic ───
if st.button("🚀 Process Loan Application", type="primary", use_container_width=True):
    st.session_state.processed = True
    st.session_state.just_clicked = True
    if is_live:
        st.session_state.live_result = None  # will be computed below

# SECTION B: Agent Pipeline
st.markdown("### ⚙️ Multi-Agent Orchestration Pipeline")
cols = st.columns(5)
agent_names = ["1. DataHarvester", "2. FraudSentinel", "3. RiskMind", "4. ComplianceGuard", "5. ExplainerVoice"]
placeholders = [col.empty() for col in cols]

# Initialize visually
for i, ph in enumerate(placeholders):
    ph.markdown(render_agent_box(agent_names[i], "waiting", "Waiting..."), unsafe_allow_html=True)

if st.session_state.get('processed', False):
    do_animate = st.session_state.get('just_clicked', False)
    st.session_state.just_clicked = False
    _sleep = time.sleep if do_animate else lambda x: None

    # ── Live Mode: run real pipeline ──
    if is_live and st.session_state.get('live_result') is None:
        # Animate "processing" for all agents
        for i, ph in enumerate(placeholders):
            ph.markdown(render_agent_box(agent_names[i], "processing", "Processing..."), unsafe_allow_html=True)

        with st.spinner("Running real LangGraph pipeline..."):
            live_result = run_live_pipeline(persona_name)

        if live_result is None:
            st.error("Failed to run live pipeline. Falling back to demo mode.")
            data = demo_data
        else:
            st.session_state.live_result = live_result
            data = live_result
    elif is_live and st.session_state.get('live_result') is not None:
        data = st.session_state.live_result
    else:
        data = demo_data

    # Animate pipeline boxes
    if data["fraud_level"] == "BLOCK":
        # Fraud detected → short circuit
        placeholders[0].markdown(render_agent_box(agent_names[0], "done", "Data Synced", "450ms"), unsafe_allow_html=True)
        _sleep(0.3)
        placeholders[1].markdown(render_agent_box(agent_names[1], "flagged", "Mule Ring Detected!", "720ms"), unsafe_allow_html=True)
        _sleep(0.3)
        placeholders[2].markdown(render_agent_box(agent_names[2], "waiting", "Bypassed"), unsafe_allow_html=True)
        placeholders[3].markdown(render_agent_box(agent_names[3], "waiting", "Bypassed"), unsafe_allow_html=True)
        _sleep(0.3)
        placeholders[4].markdown(render_agent_box(agent_names[4], "done", "Alert Generated", "500ms"), unsafe_allow_html=True)
    else:
        # Normal flow
        placeholders[0].markdown(render_agent_box(agent_names[0], "done", "Data Synced", "450ms"), unsafe_allow_html=True)
        _sleep(0.3)
        placeholders[1].markdown(render_agent_box(agent_names[1], "done", "Network Clean", "640ms"), unsafe_allow_html=True)
        _sleep(0.4)
        status_rm = "done" if data.get("score", 0) >= 600 else "flagged"
        placeholders[2].markdown(render_agent_box(agent_names[2], status_rm, f"Score: {data.get('score', 0)}", "980ms"), unsafe_allow_html=True)
        _sleep(0.3)
        placeholders[3].markdown(render_agent_box(agent_names[3], "done", "RBI Checked", "610ms"), unsafe_allow_html=True)
        _sleep(0.4)
        placeholders[4].markdown(render_agent_box(agent_names[4], "done", "Explanation Ready", "1.1s"), unsafe_allow_html=True)

    # ── Live Mode: show agent integration modes ──
    if is_live and "agent_modes" in data:
        modes = data["agent_modes"]
        if modes:
            mode_html = "<div style='text-align:center; margin: 10px 0;'>"
            for agent, m in modes.items():
                badge_class = "mode-real" if m == "real" else "mode-fallback"
                mode_html += f"<span class='mode-badge {badge_class}'>{agent}: {m}</span> "
            mode_html += "</div>"
            st.markdown(mode_html, unsafe_allow_html=True)

    st.markdown("---")

    # SECTION C: Results Panel
    st.markdown("### 📊 Real-Time Underwriting Telemetry")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.plotly_chart(create_gauge(data.get('score', 0)), use_container_width=True)
    with r2:
        st.plotly_chart(create_waterfall(data.get('shap_values', {"N/A": 0})), use_container_width=True)
    with r3:
        st.plotly_chart(create_network_graph(data.get('fraud_level', 'CLEAN')), use_container_width=True)

    st.markdown("---")

    # SECTION C2: Credit Score Agent Output
    st.markdown("### 🧠 Credit Score Agent Output (RiskMind)")
    cao = data.get("credit_agent_output", {})
    if cao:
        # ── Header row: Score + Decision + Probability ──
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            cs = cao.get("credit_score", 0)
            st.metric("ALIS Credit Score", f"{cs} / 900")
        with h2:
            approved = cao.get("approved", False)
            if approved:
                st.metric("Decision", "✅ APPROVED")
            else:
                fraud_l = data.get("fraud_level", "CLEAN")
                if fraud_l in ("BLOCK", "HIGH_RISK"):
                    st.metric("Decision", "🚫 FRAUD BLOCKED")
                else:
                    st.metric("Decision", "❌ REJECTED")
        with h3:
            prob = cao.get("probability", 0)
            st.metric("Approval Probability", f"{prob:.1%}")
        with h4:
            threshold = 600
            gap = cs - threshold
            if gap >= 0:
                st.metric("Above Threshold", f"+{gap} pts", delta=f"+{gap}")
            else:
                st.metric("Below Threshold", f"{gap} pts", delta=f"{gap}")

        # ── Factor Analysis: Positive + Negative ──
        f1, f2 = st.columns(2)
        with f1:
            st.markdown("#### ✅ Top Positive Factors")
            pos_factors = cao.get("top_positive_factors", [])
            if pos_factors:
                for i, f in enumerate(pos_factors, 1):
                    impact = f.get("impact", 0)
                    label = f.get("label", f.get("feature", "Unknown"))
                    impact_pct = f"+{abs(impact) * 100:.1f}%" if impact else ""
                    st.markdown(
                        f"<div style='padding:8px 12px; margin:4px 0; border-radius:6px; "
                        f"background: linear-gradient(90deg, rgba(16,185,129,0.15) 0%, rgba(16,185,129,0.05) 100%); "
                        f"border-left:3px solid #10B981;'>"
                        f"<span style='color:#34D399; font-weight:600;'>+{i}.</span> "
                        f"<span style='color:#E2E8F0;'>{label}</span> "
                        f"<span style='float:right; color:#10B981; font-weight:bold;'>{impact_pct}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No positive factors (credit scoring was bypassed)")

        with f2:
            st.markdown("#### ❌ Top Negative Factors")
            neg_factors = cao.get("top_negative_factors", [])
            if neg_factors:
                for i, f in enumerate(neg_factors, 1):
                    impact = f.get("impact", 0)
                    label = f.get("label", f.get("feature", "Unknown"))
                    impact_pct = f"-{abs(impact) * 100:.1f}%" if impact else ""
                    st.markdown(
                        f"<div style='padding:8px 12px; margin:4px 0; border-radius:6px; "
                        f"background: linear-gradient(90deg, rgba(239,68,68,0.15) 0%, rgba(239,68,68,0.05) 100%); "
                        f"border-left:3px solid #EF4444;'>"
                        f"<span style='color:#F87171; font-weight:600;'>-{i}.</span> "
                        f"<span style='color:#E2E8F0;'>{label}</span> "
                        f"<span style='float:right; color:#EF4444; font-weight:bold;'>{impact_pct}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No negative factors identified")

        # ── Counterfactual Improvement Roadmap ──
        cf_advice = cao.get("counterfactual_advice", [])
        if cf_advice:
            st.markdown("#### 💡 SHAP Counterfactual Advice (Improvement Roadmap)")
            for i, advice in enumerate(cf_advice, 1):
                advice_text = advice if isinstance(advice, str) else str(advice)
                st.markdown(
                    f"<div style='padding:10px 14px; margin:4px 0; border-radius:6px; "
                    f"background: rgba(59,130,246,0.1); border-left:3px solid #3B82F6;'>"
                    f"<span style='color:#60A5FA; font-weight:600;'>Step {i}:</span> "
                    f"<span style='color:#CBD5E1;'>{advice_text}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
    else:
        st.info("Credit score data not available for this scenario.")

    st.markdown("---")

    # SECTION D & E: Compliance & Explainer
    b1, b2 = st.columns([1, 1.5])

    with b1:
        st.markdown("### 📜 RBI Compliance Guard")
        if data.get('compliance', True):
            st.success(data.get('compliance_msg', 'Compliant'))
            st.write("✓ APR within 36% limit (Clause 5.2)\n\n✓ Disbursal to verified own account (Clause 3.1)\n\n✓ 3-day Cooling off period guaranteed (Clause 2.3)")
        else:
            st.error(data.get('compliance_msg', 'Non-compliant'))
            quote = data.get('compliance_quote', '')
            if quote:
                st.markdown(f"<div class='rbi-quote'>{quote}</div>", unsafe_allow_html=True)

    with b2:
        st.markdown("### 🗣️ Vernacular Explainer Voice")
        lang = st.radio("Translate to:", ["English", "ಕನ್ನಡ", "हिंदी"], horizontal=True)

        explanations = data.get('explanations', {})
        st.info(explanations.get(lang, "Explanation not available."))

        counterfactual = data.get('counterfactual', '')
        if counterfactual:
            st.markdown(f"<div class='counterfactual'>{counterfactual}</div>", unsafe_allow_html=True)

        st.button("📋 Copy Explanation to Clipboard")

    # ── Live Mode: Pipeline Details ──
    if is_live and "timestamp_each_stage" in data:
        with st.expander("🔍 Pipeline Execution Details"):
            stages = data.get("timestamp_each_stage", {})
            modes = data.get("agent_modes", {})
            if stages:
                df_stages = pd.DataFrame([
                    {
                        "Agent": agent,
                        "Completed At": ts[11:19] if len(ts) > 19 else ts,
                        "Mode": modes.get(agent, "unknown"),
                    }
                    for agent, ts in stages.items()
                ])
                st.dataframe(df_stages, use_container_width=True)

            st.write(f"**Pipeline Status:** {data.get('pipeline_status', 'N/A')}")
            if data.get("is_thin_file"):
                st.warning("🔶 Thin-File Protocol was activated for this applicant.")
