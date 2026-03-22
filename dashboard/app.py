"""
ALIS — Agentic Loan Intelligence System
Competition Demo Dashboard (RVCE FinTech Summit 2026)

Run:
  streamlit run app.py
"""

import time
import random
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

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
        # Fraud logic: color central nodes red if flagged
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
mode = st.sidebar.radio("Backend Mode", ["Demo Mode (Pre-computed)", "Live Mode (API calling)"])
st.sidebar.markdown("---")
st.sidebar.subheader("Select Applicant Scenario:")

persona_name = st.sidebar.radio("Evaluate:", list(PERSONAS.keys()))
data = PERSONAS[persona_name]

st.sidebar.markdown(f"**Profile:**<br>{data['profile']}", unsafe_allow_html=True)
st.sidebar.markdown(f"**Requested Loan:** ₹{data['amount']:,}")

if st.sidebar.button("🔄 Reset / Start Fresh", use_container_width=True):
    # Streamlit hack to rerun and clear state
    st.rerun()

# ─── Main Logic ───
run_demo = st.button("🚀 Process Loan Application", type="primary", use_container_width=True)

# SECTION B: Agent Pipeline
st.markdown("### ⚙️ Multi-Agent Orchestration Pipeline")
cols = st.columns(5)
agent_names = ["1. DataHarvester", "2. FraudSentinel", "3. RiskMind", "4. ComplianceGuard", "5. ExplainerVoice"]
placeholders = [col.empty() for col in cols]

# Initialize visually
for i, ph in enumerate(placeholders):
    ph.markdown(render_agent_box(agent_names[i], "waiting", "Waiting..."), unsafe_allow_html=True)

if run_demo:
    # Animate pipeline
    agents_status = ["waiting"] * 5
    
    # 1. Data Harvester
    placeholders[0].markdown(render_agent_box(agent_names[0], "processing", "Collecting UPI Data...", ""), unsafe_allow_html=True)
    time.sleep(0.5)
    placeholders[0].markdown(render_agent_box(agent_names[0], "done", "Data Synced", "450ms"), unsafe_allow_html=True)
    
    # 2. Fraud Sentinel
    placeholders[1].markdown(render_agent_box(agent_names[1], "processing", "Building Network...", ""), unsafe_allow_html=True)
    time.sleep(0.8)
    if data["fraud_level"] == "BLOCK":
        placeholders[1].markdown(render_agent_box(agent_names[1], "flagged", "Mule Ring Detected!", "720ms"), unsafe_allow_html=True)
        # Short circuit logic!
        placeholders[2].markdown(render_agent_box(agent_names[2], "waiting", "Bypassed"), unsafe_allow_html=True)
        placeholders[3].markdown(render_agent_box(agent_names[3], "waiting", "Bypassed"), unsafe_allow_html=True)
        placeholders[4].markdown(render_agent_box(agent_names[4], "processing", "Generating Alert...", ""), unsafe_allow_html=True)
        time.sleep(0.6)
        placeholders[4].markdown(render_agent_box(agent_names[4], "done", "Explanation Ready", "500ms"), unsafe_allow_html=True)
    else:
        placeholders[1].markdown(render_agent_box(agent_names[1], "done", "Network Clean", "640ms"), unsafe_allow_html=True)
        
        # 3. Risk Mind
        placeholders[2].markdown(render_agent_box(agent_names[2], "processing", "Scoring & SHAP...", ""), unsafe_allow_html=True)
        time.sleep(1.0)
        status_rm = "done" if data["decision"] == "APPROVED" else "flagged"
        placeholders[2].markdown(render_agent_box(agent_names[2], status_rm, f"Score: {data['score']}", "980ms"), unsafe_allow_html=True)
        
        # 4. Compliance Guard
        placeholders[3].markdown(render_agent_box(agent_names[3], "processing", "RAG Check: RBI...", ""), unsafe_allow_html=True)
        time.sleep(0.7)
        placeholders[3].markdown(render_agent_box(agent_names[3], "done", "Compliant", "610ms"), unsafe_allow_html=True)
        
        # 5. Explainer Voice
        placeholders[4].markdown(render_agent_box(agent_names[4], "processing", "Translating (LLM)...", ""), unsafe_allow_html=True)
        time.sleep(1.2)
        placeholders[4].markdown(render_agent_box(agent_names[4], "done", "Explanation Ready", "1.1s"), unsafe_allow_html=True)

    st.markdown("---")

    # SECTION C: Results Panel
    st.markdown("### 📊 Real-Time Underwriting Telemetry")    
    r1, r2, r3 = st.columns(3)
    with r1:
        st.plotly_chart(create_gauge(data['score']), use_container_width=True)
    with r2:
        st.plotly_chart(create_waterfall(data['shap_values']), use_container_width=True)
    with r3:
        st.plotly_chart(create_network_graph(data['fraud_level']), use_container_width=True)

    st.markdown("---")
    
    # SECTION D & E: Compliance & Explainer
    b1, b2 = st.columns([1, 1.5])
    
    with b1:
        st.markdown("### 📜 RBI Compliance Guard")
        if data['compliance']:
            st.success(data['compliance_msg'])
            st.write("✓ APR within 36% limit (Clause 5.2)\n\n✓ Disbursal to verified own account (Clause 3.1)\n\n✓ 3-day Cooling off period guaranteed (Clause 2.3)")
        else:
            st.error(data['compliance_msg'])
            st.markdown(f"<div class='rbi-quote'>{data['compliance_quote']}</div>", unsafe_allow_html=True)

    with b2:
        st.markdown("### 🗣️ Vernacular Explainer Voice")
        lang = st.radio("Translate to:", ["English", "ಕನ್ನಡ", "हिंदी"], horizontal=True)
        
        # Render the text
        st.info(data['explanations'][lang])
        
        # Show counterfactual cleanly
        st.markdown(f"<div class='counterfactual'>{data['counterfactual']}</div>", unsafe_allow_html=True)
        
        # Bonus UX
        st.button("📋 Copy Explanation to Clipboard")
