"""
ALIS — FraudSentinel: FastAPI Fraud Assessment Endpoint
========================================================
REST API for real-time fraud risk assessment of loan applicants.

Endpoints:
    POST /fraud-check    → Full fraud risk assessment
    GET  /health         → Service health check
    GET  /graph-stats    → Transaction graph statistics

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8001 --reload
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))

from graph_builder import NODE_FEATURE_DESCRIPTIONS
from scorer import FraudScorer

# ─── Pydantic Schemas ────────────────────────────────────────────────────────


class FraudCheckRequest(BaseModel):
    """Input: applicant node ID in the transaction graph."""
    applicant_id: str = Field(
        ..., description="Applicant's UPI node ID in the transaction graph"
    )

    model_config = {"json_schema_extra": {
        "examples": [{"applicant_id": "L_0010"}]
    }}


class SuspiciousAccount(BaseModel):
    account_id: str
    node_type: str
    fraud_probability: float
    account_age_days: int
    connection_type: str
    txn_amount: float


class FraudCheckResponse(BaseModel):
    """Output: full fraud risk assessment."""
    applicant_id: str
    fraud_risk_score: int = Field(..., ge=0, le=100)
    risk_level: str = Field(..., description="CLEAN | CAUTION | HIGH_RISK | BLOCK")
    gnn_probability: float | None = Field(None, description="Raw GraphSAGE fraud probability")
    explanation: list[str] = Field(..., description="Human-readable risk factors")
    connected_suspicious_accounts: list[SuspiciousAccount]
    node_features: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    graph_nodes: int
    graph_edges: int


class GraphStatsResponse(BaseModel):
    total_nodes: int
    total_edges: int
    legitimate_nodes: int
    mule_nodes: int
    fraudster_nodes: int
    fraud_edges: int


# ─── App Lifecycle ───────────────────────────────────────────────────────────

scorer_instance: FraudScorer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scorer_instance
    try:
        scorer_instance = FraudScorer()
        print("  ✓ FraudSentinel model + graph loaded")
    except FileNotFoundError as e:
        print(f"  ✗ Model not found: {e}")
        print("  → Run 'python model.py' first to train.")
        scorer_instance = None
    yield
    scorer_instance = None


# ─── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="ALIS FraudSentinel API",
    description=(
        "Graph-based fraud detection for loan applicants. "
        "Uses GraphSAGE on UPI transaction networks to detect "
        "synthetic identities, mule accounts, and fraud rings."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    if scorer_instance:
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            graph_nodes=scorer_instance.G.number_of_nodes(),
            graph_edges=scorer_instance.G.number_of_edges(),
        )
    return HealthResponse(status="degraded", model_loaded=False, graph_nodes=0, graph_edges=0)


@app.get("/graph-stats", response_model=GraphStatsResponse, tags=["Analytics"])
async def graph_stats():
    if not scorer_instance:
        raise HTTPException(503, "Model not loaded")

    G = scorer_instance.G
    return GraphStatsResponse(
        total_nodes=G.number_of_nodes(),
        total_edges=G.number_of_edges(),
        legitimate_nodes=sum(1 for _, d in G.nodes(data=True)
                            if d.get("node_type") == "legitimate"),
        mule_nodes=sum(1 for _, d in G.nodes(data=True)
                       if d.get("node_type") == "mule"),
        fraudster_nodes=sum(1 for _, d in G.nodes(data=True)
                            if d.get("node_type") == "fraudster"),
        fraud_edges=sum(1 for _, _, d in G.edges(data=True)
                        if d.get("is_fraud_edge")),
    )


@app.post("/fraud-check", response_model=FraudCheckResponse, tags=["Fraud Detection"])
async def check_fraud(request: FraudCheckRequest):
    """
    Score an applicant for fraud risk using GraphSAGE + structural heuristics.

    Returns fraud_risk_score (0-100), risk_level, human-readable explanations,
    and a list of connected suspicious accounts.
    """
    if not scorer_instance:
        raise HTTPException(503, "Model not loaded. Run 'python model.py' first.")

    try:
        result = scorer_instance.score_applicant(request.applicant_id)

        return FraudCheckResponse(
            applicant_id=request.applicant_id,
            fraud_risk_score=result["fraud_risk_score"],
            risk_level=result["risk_level"],
            gnn_probability=result["gnn_probability"],
            explanation=result["explanation"],
            connected_suspicious_accounts=[
                SuspiciousAccount(**acc)
                for acc in result["connected_suspicious_accounts"]
            ],
            node_features=result["node_features"],
        )
    except Exception as e:
        raise HTTPException(500, f"Scoring error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
