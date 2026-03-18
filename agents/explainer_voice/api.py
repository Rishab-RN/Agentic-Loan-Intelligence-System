"""
ALIS — ExplainerVoice: FastAPI Vernacular Explanation Endpoint
===============================================================
REST API for generating warm, multilingual loan decision explanations.

Endpoints:
    POST /explain           → Full vernacular explanation
    POST /translate-shap    → Translate SHAP values to language
    POST /counterfactual    → Improvement roadmap
    GET  /health            → Health check
    GET  /languages         → Supported languages

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8003 --reload
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))

from llm_engine import ExplainerLLM
from shap_translator import SUPPORTED_LANGUAGES, translate_shap_to_language
from templates import build_explanation, generate_counterfactual

# ─── Pydantic Schemas ────────────────────────────────────────────────────────


class DecisionInput(BaseModel):
    """Full decision packet from the pipeline."""
    applicant_name: str = Field(default="Applicant")
    decision: str = Field(
        ..., description="APPROVED | REJECTED | MORE_INFO_NEEDED | FRAUD_FLAGGED"
    )
    credit_score: int = Field(..., ge=0, le=900)
    loan_amount: int = Field(..., ge=0)
    shap_values: dict[str, float] = Field(
        default_factory=dict, description="Feature name → SHAP value"
    )
    fraud_risk_level: str = Field(default="CLEAN")
    compliance_status: bool = Field(default=True)
    missing_items: list[str] = Field(default_factory=list)
    cooling_off_days: int = Field(default=3)
    days_to_improve: int = Field(default=45)
    helpline: str = Field(default="1800-XXX-XXXX")
    language: str = Field(default="english", description="english | kannada | hindi")
    use_llm: bool = Field(
        default=True, description="Use Ollama LLM for natural polish (if available)"
    )

    model_config = {"json_schema_extra": {
        "examples": [{
            "applicant_name": "Ramu",
            "decision": "REJECTED",
            "credit_score": 510,
            "loan_amount": 25000,
            "shap_values": {
                "utility_bill_payment_consistency": -0.18,
                "savings_behavior_score": -0.15,
                "upi_txn_frequency_30d": 0.12,
            },
            "language": "kannada",
        }]
    }}


class ExplanationResponse(BaseModel):
    applicant_name: str
    decision: str
    credit_score: int
    language: str
    explanation: str
    mode: str = Field(..., description="'template' or 'llm-refined'")


class ShapTranslateInput(BaseModel):
    shap_values: dict[str, float]
    language: str = "english"


class ShapTranslation(BaseModel):
    feature: str
    name: str
    shap_value: float
    direction: str
    explanation: str
    advice: str = ""


class CounterfactualInput(BaseModel):
    current_score: int = Field(..., ge=0, le=900)
    target_score: int = Field(default=650, ge=0, le=900)
    shap_values: dict[str, float]
    language: str = "english"


class CounterfactualStep(BaseModel):
    feature: str
    improvement: str
    estimated_score_gain: int
    days_needed: int
    priority: int
    cumulative_score: int


class HealthResponse(BaseModel):
    status: str
    ollama_available: bool
    llm_model: str
    supported_languages: list[str]


# ─── App Lifecycle ───────────────────────────────────────────────────────────

llm_engine: ExplainerLLM | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_engine
    llm_engine = ExplainerLLM()
    mode = "LLM-available" if llm_engine.available else "template-only"
    print(f"  ✓ ExplainerVoice initialized ({mode})")
    yield
    llm_engine = None


# ─── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="ALIS ExplainerVoice API",
    description=(
        "Multilingual vernacular explanations for loan decisions. "
        "Generates warm, human-readable explanations in Kannada, Hindi, "
        "and English using template engine + optional Ollama LLM refinement."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status="healthy",
        ollama_available=llm_engine.available if llm_engine else False,
        llm_model=llm_engine.model if llm_engine else "N/A",
        supported_languages=sorted(SUPPORTED_LANGUAGES),
    )


@app.get("/languages", tags=["Info"])
async def get_languages():
    return {
        "supported": sorted(SUPPORTED_LANGUAGES),
        "default": "english",
        "note": "Kannada and Hindi translations are hand-crafted for Tier-2 city comprehension",
    }


@app.post("/explain", response_model=ExplanationResponse, tags=["Explanation"])
async def generate_explanation(input_data: DecisionInput):
    """
    Generate a warm, vernacular explanation for a loan decision.

    Accepts the full decision packet from the ALIS pipeline (RiskMind
    score + FraudSentinel check + ComplianceGuard result) and returns
    a human-readable explanation in the chosen language.
    """
    if not llm_engine:
        raise HTTPException(503, "ExplainerVoice not initialized")

    decision_data = input_data.model_dump()
    language = decision_data.pop("language", "english")
    use_llm = decision_data.pop("use_llm", True)

    if use_llm and llm_engine.available:
        explanation = llm_engine.generate_vernacular_explanation(
            decision_data, language
        )
        mode = "llm-refined"
    else:
        explanation = build_explanation(decision_data, language)
        mode = "template"

    return ExplanationResponse(
        applicant_name=input_data.applicant_name,
        decision=input_data.decision,
        credit_score=input_data.credit_score,
        language=language,
        explanation=explanation,
        mode=mode,
    )


@app.post(
    "/translate-shap",
    response_model=list[ShapTranslation],
    tags=["SHAP"],
)
async def translate_shap(input_data: ShapTranslateInput):
    """Translate SHAP feature attributions to human-readable language."""
    results = translate_shap_to_language(
        input_data.shap_values, input_data.language
    )
    return [ShapTranslation(**r) for r in results]


@app.post(
    "/counterfactual",
    response_model=list[CounterfactualStep],
    tags=["Counterfactual"],
)
async def get_counterfactual(input_data: CounterfactualInput):
    """Generate a SHAP-grounded improvement roadmap."""
    results = generate_counterfactual(
        input_data.current_score,
        input_data.target_score,
        input_data.shap_values,
        input_data.language,
    )
    return [CounterfactualStep(**r) for r in results]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8003, reload=True)
