<p align="center">
  <h1 align="center">🧠 ALIS — Agentic Loan Intelligence System</h1>
  <p align="center">
    <strong>AI-powered credit underwriting for India's financially excluded population</strong><br>
    <em>Multi-agent system using alternative data (UPI, utility bills, mobile patterns) instead of CIBIL scores</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python" alt="Python">
    <img src="https://img.shields.io/badge/LangGraph-Multi--Agent-orange?logo=langchain" alt="LangGraph">
    <img src="https://img.shields.io/badge/XGBoost-SHAP-green" alt="XGBoost">
    <img src="https://img.shields.io/badge/PyG-GraphSAGE-red" alt="PyG">
    <img src="https://img.shields.io/badge/Fairlearn-RBI%20FREE--AI-purple" alt="Fairlearn">
    <img src="https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?logo=streamlit" alt="Streamlit">
  </p>
</p>

---

## 🎯 Problem Statement

**800 million Indians** lack formal credit histories (CIBIL scores). Gig workers, auto drivers, kirana shop owners, and domestic workers in Tier-2/3 cities are **creditworthy by behavior** but **invisible to banks**. ALIS solves this by scoring creditworthiness from alternative digital footprints — UPI transactions, utility bill payments, mobile recharge patterns — while detecting fraud at application time and explaining every decision in **Kannada/Hindi**.

## 🏗️ System Architecture

ALIS is a **6-agent pipeline** orchestrated by LangGraph:

```
┌──────────────┐    ┌────────────────┐    ┌──────────┐    ┌─────────────────┐    ┌────────────────┐
│ DataHarvester│───▶│ FraudSentinel  │───▶│ RiskMind │───▶│ ComplianceGuard │───▶│ ExplainerVoice │
│  (Validate)  │    │ (Graph Fraud)  │    │ (Score)  │    │   (RBI Rules)   │    │  (Vernacular)  │
└──────────────┘    └────────────────┘    └──────────┘    └─────────────────┘    └────────────────┘
                           │                                                              │
                    ❌ BLOCK ──────────────────────── Short-circuit ──────────────────────▶│
```

| Agent | Tech Stack | What It Does |
|-------|-----------|--------------|
| **RiskMind** | XGBoost + SHAP | Credit scoring using 12 alternative data features. AUC-ROC: **0.90** |
| **FraudSentinel** | PyTorch Geometric (GraphSAGE) | Detects synthetic identity fraud, mule accounts, and fraud rings from UPI transaction networks at application time (t=0) |
| **ComplianceGuard** | ChromaDB RAG + Deterministic Rules | Checks every loan offer against 6 RBI Digital Lending Guidelines with auto-correction for fixable violations |
| **ExplainerVoice** | Template Engine + Ollama LLM | Generates warm, human-readable explanations in **English**, **ಕನ್ನಡ (Kannada)**, **हिंदी (Hindi)** with SHAP-grounded counterfactual advice |
| **LoanOrchestrator** | LangGraph StateGraph | Coordinates all agents with conditional routing: fraud short-circuiting, thin-file protocol, human escalation |
| **Fairness Audit** | Microsoft Fairlearn | RBI FREE-AI compliant bias detection and mitigation with ExponentiatedGradient |

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Git

### Setup
```bash
# Clone the repository
git clone https://github.com/Rishab-RN/Agentic-Loan-Intelligence-System.git
cd Agentic-Loan-Intelligence-System

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install all dependencies
pip install -r agents/risk_mind/requirements.txt
pip install -r agents/fraud_sentinel/requirements.txt
pip install -r agents/compliance_guard/requirements.txt
pip install -r agents/explainer_voice/requirements.txt
pip install -r agents/loan_orchestrator/requirements.txt
pip install -r agents/fairness_audit/requirements.txt
pip install -r dashboard/requirements.txt
```

### Train Models & Generate Artifacts
```bash
# 1. RiskMind — Train credit scoring model
python agents/risk_mind/data_generator.py
python agents/risk_mind/train.py

# 2. FraudSentinel — Build graph and train GNN
python agents/fraud_sentinel/graph_builder.py
python agents/fraud_sentinel/model.py

# 3. ComplianceGuard — Index RBI guidelines
python agents/compliance_guard/document_loader.py

# 4. Fairness Audit — Run bias audit and generate PDF report
python agents/fairness_audit/audit.py
python agents/fairness_audit/report_generator.py
```

### Run the Demo Dashboard
```bash
streamlit run dashboard/app.py
```

### Run the Multi-Agent Pipeline
```bash
python agents/loan_orchestrator/orchestrator.py
```

### Run Individual Agent APIs
```bash
python agents/risk_mind/api.py              # Port 8000
python agents/fraud_sentinel/api.py         # Port 8001
python agents/compliance_guard/api.py       # Port 8002
python agents/explainer_voice/api.py        # Port 8003
```

## 📊 Key Results

| Metric | Value |
|--------|-------|
| RiskMind AUC-ROC | **0.90** (5-fold CV: 0.89 ± 0.008) |
| FraudSentinel Accuracy | **100%** on synthetic graph |
| ComplianceGuard | **6/6** RBI rules enforced with auto-correction |
| Fairness Audit | Demographic Parity gap **mitigated** via ExponentiatedGradient |
| Languages Supported | English, ಕನ್ನಡ (Kannada), हिंदी (Hindi) |

## 🏛️ RBI Compliance

ALIS is aligned with:
- **RBI Digital Lending Guidelines** (Sept 2022, Updated 2023) — APR disclosure, KYC verification, cooling-off period, disbursal rules, recovery agent hours, credit limit consent
- **RBI FREE-AI Framework** (2025) — Fairlearn-based algorithmic fairness audit with automated bias mitigation

## 📁 Project Structure

```
ALIS/
├── agents/
│   ├── risk_mind/           # Credit scoring (XGBoost + SHAP)
│   │   ├── data_generator.py
│   │   ├── train.py
│   │   ├── explainer.py
│   │   └── api.py
│   ├── fraud_sentinel/      # Graph fraud detection (GraphSAGE)
│   │   ├── graph_builder.py
│   │   ├── model.py
│   │   ├── scorer.py
│   │   ├── visualize.py
│   │   └── api.py
│   ├── compliance_guard/    # RBI compliance (RAG + Rules)
│   │   ├── document_loader.py
│   │   ├── rag_engine.py
│   │   ├── compliance_checker.py
│   │   ├── audit_logger.py
│   │   └── api.py
│   ├── explainer_voice/     # Vernacular explanations
│   │   ├── shap_translator.py
│   │   ├── templates.py
│   │   ├── llm_engine.py
│   │   └── api.py
│   ├── loan_orchestrator/   # LangGraph multi-agent coordinator
│   │   └── orchestrator.py
│   └── fairness_audit/      # Fairlearn bias audit
│       ├── data_generator.py
│       ├── audit.py
│       └── report_generator.py
├── dashboard/
│   └── app.py               # Streamlit competition demo
├── docs/
│   └── ALIS_SYSTEM_ARCHITECTURE.md
└── README.md
```

## 💡 Key Innovations

1. **Alternative Data Credit Scoring** — First system to use UPI transaction patterns, utility bill consistency, and mobile recharge regularity as credit features for India's unbanked population
2. **Application-Time Fraud Detection** — Graph Neural Network (GraphSAGE) detects mule accounts and fraud rings at loan application time (t=0), not post-disbursal
3. **Regulatory-Grade Compliance** — RAG-powered RBI guideline checker with deterministic rule enforcement and auto-correction
4. **Vernacular Explainability** — SHAP-grounded explanations translated to Kannada and Hindi with actionable counterfactual advice
5. **Algorithmic Fairness** — Microsoft Fairlearn audit with automated bias mitigation, aligned with RBI FREE-AI framework

## 👥 Team

Built by students at **RV College of Engineering (RVCE), Bengaluru** for the **RVCE FinTech Innovation Summit 2026**.

## 📄 License

This project is for academic and competition purposes. All rights reserved.
