"""
ALIS — RiskMind: FastAPI Credit Scoring Endpoint
==================================================
Production-ready REST API for real-time credit decisions.

Endpoints:
    POST /score     →  Credit score + explanation + counterfactual advice
    GET  /health    →  Service health check
    GET  /features  →  List of required input features with descriptions

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure the agent package is importable
sys.path.insert(0, str(Path(__file__).parent))

from data_generator import FEATURE_COLUMNS, FEATURE_DESCRIPTIONS
from explainer import RiskMindExplainer

# ─── Pydantic Schemas ────────────────────────────────────────────────────────


class ApplicantInput(BaseModel):
    """Input schema — all 12 alternative-data features for one applicant."""
    upi_txn_frequency_30d: float = Field(
        ..., ge=0, description="Average daily UPI transactions in last 30 days"
    )
    upi_merchant_diversity_score: float = Field(
        ..., ge=0, le=1, description="Unique merchant categories / total transactions"
    )
    utility_bill_payment_consistency: float = Field(
        ..., ge=0, le=1, description="Fraction of bills paid on time (12 months)"
    )
    mobile_recharge_regularity: float = Field(
        ..., gt=0, description="Average days between recharges (lower = more regular)"
    )
    income_estimate_monthly: float = Field(
        ..., ge=0, description="Estimated monthly income from UPI credits (INR)"
    )
    income_volatility_cv: float = Field(
        ..., ge=0, description="Coefficient of variation of monthly income"
    )
    bnpl_outstanding_ratio: float = Field(
        ..., ge=0, description="BNPL outstanding balance / monthly income"
    )
    multi_loan_app_count: int = Field(
        ..., ge=0, description="Number of loan apps installed on device"
    )
    evening_txn_ratio: float = Field(
        ..., ge=0, le=1, description="Fraction of transactions after 8 PM"
    )
    savings_behavior_score: float = Field(
        ..., ge=0, le=1, description="Savings transfer regularity score"
    )
    peer_transfer_reciprocity: float = Field(
        ..., ge=0, description="Ratio of sent/received peer transfers"
    )
    device_tenure_months: float = Field(
        ..., ge=0, description="Months using current device"
    )

    model_config = {"json_schema_extra": {
        "examples": [{
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
        }]
    }}


class FactorExplanation(BaseModel):
    feature: str
    description: str
    value: float
    shap_impact: float
    explanation: str


class CounterfactualAdvice(BaseModel):
    feature: str
    current_value: float
    target_value: float
    advice: str
    estimated_score_improvement: int
    projected_score: int


class CreditDecisionResponse(BaseModel):
    """Output schema — full credit decision with explanations."""
    credit_score: int = Field(..., ge=0, le=900, description="Credit score 0-900")
    approved: bool = Field(..., description="Whether the applicant is approved")
    approval_threshold: int = Field(default=500)
    probability: float = Field(..., description="Calibrated approval probability")
    top_positive_factors: list[FactorExplanation]
    top_negative_factors: list[FactorExplanation]
    counterfactual_advice: list[CounterfactualAdvice]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class FeatureInfo(BaseModel):
    name: str
    description: str


# ─── App Lifecycle ───────────────────────────────────────────────────────────

explainer_instance: RiskMindExplainer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup."""
    global explainer_instance
    try:
        explainer_instance = RiskMindExplainer()
        print("  ✓ RiskMind model loaded successfully")
    except FileNotFoundError as e:
        print(f"  ✗ Model files not found: {e}")
        print("  → Run 'python train.py' first to train the model.")
        explainer_instance = None
    yield
    explainer_instance = None


# ─── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="ALIS RiskMind API",
    description=(
        "Credit scoring engine for India's gig economy workers. "
        "Uses alternative data (UPI, utility bills, mobile recharges) "
        "instead of CIBIL scores to assess creditworthiness."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Service health check — verifies model is loaded."""
    return HealthResponse(
        status="healthy" if explainer_instance else "degraded",
        model_loaded=explainer_instance is not None,
        version="1.0.0",
    )


@app.get("/features", response_model=list[FeatureInfo], tags=["Documentation"])
async def list_features():
    """List all required input features with descriptions."""
    return [
        FeatureInfo(name=f, description=FEATURE_DESCRIPTIONS.get(f, f))
        for f in FEATURE_COLUMNS
    ]


@app.post("/score", response_model=CreditDecisionResponse, tags=["Scoring"])
async def score_applicant(applicant: ApplicantInput):
    """
    Score an applicant and return credit decision with full SHAP explanation.

    Accepts 12 alternative-data features and returns:
    - Credit score (0-900)
    - Approval decision (threshold: 500)
    - Top 5 positive factors with plain-English explanations
    - Top 3 negative factors with plain-English explanations
    - Counterfactual advice (actionable steps to improve score)
    """
    if explainer_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'python train.py' first.",
        )

    try:
        features_dict = applicant.model_dump()
        result = explainer_instance.explain_decision(features_dict)

        return CreditDecisionResponse(
            credit_score=result["credit_score"],
            approved=result["approved"],
            approval_threshold=result["approval_threshold"],
            probability=result["probability"],
            top_positive_factors=[
                FactorExplanation(**f) for f in result["top_positive_factors"]
            ],
            top_negative_factors=[
                FactorExplanation(**f) for f in result["top_negative_factors"]
            ],
            counterfactual_advice=[
                CounterfactualAdvice(**cf) for cf in result["counterfactual_advice"]
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")


# ─── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
