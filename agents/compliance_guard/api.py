"""
ALIS — ComplianceGuard: FastAPI Compliance Endpoint
=====================================================
REST API for real-time RBI compliance verification of loan offers.

Endpoints:
    POST /compliance-check    → Full compliance report
    POST /query-guidelines    → Query RBI guidelines (RAG)
    GET  /health              → Service health check
    GET  /audit-stats         → Audit trail statistics

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8002 --reload
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))

from audit_logger import AuditLogger
from compliance_checker import auto_adjust_terms, check_loan_compliance
from rag_engine import RBIQueryEngine

# ─── Pydantic Schemas ────────────────────────────────────────────────────────


class RecoveryHours(BaseModel):
    start: int = Field(..., ge=0, le=23, description="Start hour (24h format)")
    end: int = Field(..., ge=0, le=23, description="End hour (24h format)")


class LoanOfferInput(BaseModel):
    """Loan offer to check for RBI compliance."""
    applicant_id: str = Field(default="anonymous", description="Applicant identifier")
    apr: float = Field(..., ge=0, description="Annual Percentage Rate (%)")
    disbursal_account_type: str = Field(
        ..., description="'own' or 'third_party'"
    )
    kyc_completed: bool = Field(..., description="KYC verified before disbursal")
    credit_limit_auto_increase: bool = Field(
        ..., description="Auto credit limit increase enabled"
    )
    cooling_off_days: int = Field(..., ge=0, description="Cooling-off period (days)")
    recovery_contact_hours: RecoveryHours = Field(
        ..., description="Recovery agent contact window"
    )

    model_config = {"json_schema_extra": {
        "examples": [{
            "applicant_id": "APP_001",
            "apr": 24.0,
            "disbursal_account_type": "own",
            "kyc_completed": True,
            "credit_limit_auto_increase": False,
            "cooling_off_days": 5,
            "recovery_contact_hours": {"start": 9, "end": 18},
        }]
    }}


class ViolationDetail(BaseModel):
    rule: str
    severity: str
    rbi_clause: str
    field: str = ""
    explanation: str
    correction: str = ""


class ComplianceResponse(BaseModel):
    """Full compliance check result."""
    applicant_id: str
    is_compliant: bool
    has_critical_violations: bool
    violation_count: int
    violations: list[dict]
    compliance_report: str
    recommended_corrections: list[str]
    offer_hash: str
    audit_id: int = Field(..., description="Audit trail row ID")


class AutoCorrectionResponse(BaseModel):
    """Result of auto-correction attempt."""
    original_violations: int
    adjusted_offer: dict
    adjustments_made: list[str]
    post_adjustment_violations: int
    post_adjustment_compliant: bool


class GuidelineQueryInput(BaseModel):
    question: str = Field(..., description="Compliance question to query")
    n_results: int = Field(default=3, ge=1, le=10)


class GuidelineResult(BaseModel):
    text: str
    clause: str
    section: str
    page: int
    relevance_score: float


class HealthResponse(BaseModel):
    status: str
    rag_available: bool
    audit_db_available: bool
    total_audits: int


class AuditStatsResponse(BaseModel):
    total_checks: int
    compliant: int
    non_compliant: int
    hard_blocked: int
    compliance_rate: float


# ─── App Lifecycle ───────────────────────────────────────────────────────────

rag_engine: RBIQueryEngine | None = None
audit_logger: AuditLogger | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine, audit_logger
    try:
        rag_engine = RBIQueryEngine()
        print(f"  ✓ RAG engine: {'loaded' if rag_engine.available else 'unavailable'}")
    except Exception:
        rag_engine = None

    audit_logger = AuditLogger()
    print("  ✓ Audit logger initialized")
    yield
    rag_engine = None
    audit_logger = None


# ─── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="ALIS ComplianceGuard API",
    description=(
        "RBI Digital Lending Guidelines compliance verification. "
        "Checks loan offers against regulatory requirements using "
        "deterministic rules + RAG semantic checks."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    stats = audit_logger.get_stats() if audit_logger else {"total_checks": 0}
    return HealthResponse(
        status="healthy",
        rag_available=rag_engine.available if rag_engine else False,
        audit_db_available=audit_logger is not None,
        total_audits=stats["total_checks"],
    )


@app.get("/audit-stats", response_model=AuditStatsResponse, tags=["Audit"])
async def get_audit_stats():
    if not audit_logger:
        raise HTTPException(503, "Audit logger not initialized")
    stats = audit_logger.get_stats()
    return AuditStatsResponse(**stats)


@app.post("/compliance-check", response_model=ComplianceResponse, tags=["Compliance"])
async def check_compliance(offer: LoanOfferInput):
    """
    Check a loan offer against RBI Digital Lending Guidelines.

    Returns compliance status, all violations with RBI clause references,
    severity levels, and recommended corrections. Every check is logged
    to an immutable audit trail.
    """
    # Convert Pydantic model to dict for checker
    offer_dict = {
        "apr": offer.apr,
        "disbursal_account_type": offer.disbursal_account_type,
        "kyc_completed": offer.kyc_completed,
        "credit_limit_auto_increase": offer.credit_limit_auto_increase,
        "cooling_off_days": offer.cooling_off_days,
        "recovery_contact_hours": {
            "start": offer.recovery_contact_hours.start,
            "end": offer.recovery_contact_hours.end,
        },
    }

    use_rag = rag_engine is not None and rag_engine.available
    result = check_loan_compliance(offer_dict, use_rag=use_rag)

    # Log to audit trail
    audit_id = 0
    if audit_logger:
        audit_id = audit_logger.log_check(
            applicant_id=offer.applicant_id,
            loan_offer=offer_dict,
            check_result=result,
        )

    return ComplianceResponse(
        applicant_id=offer.applicant_id,
        is_compliant=result["is_compliant"],
        has_critical_violations=result["has_critical_violations"],
        violation_count=result["violation_count"],
        violations=result["violations"],
        compliance_report=result["compliance_report"],
        recommended_corrections=result["recommended_corrections"],
        offer_hash=result["offer_hash"],
        audit_id=audit_id,
    )


@app.post(
    "/compliance-check/auto-correct",
    response_model=AutoCorrectionResponse,
    tags=["Compliance"],
)
async def auto_correct_compliance(offer: LoanOfferInput):
    """
    Check compliance and auto-correct fixable violations.

    This endpoint simulates what the LoanOrchestrator does:
    1. Run compliance check
    2. If non-critical violations found, auto-adjust terms
    3. Re-run compliance check on adjusted terms
    """
    offer_dict = {
        "apr": offer.apr,
        "disbursal_account_type": offer.disbursal_account_type,
        "kyc_completed": offer.kyc_completed,
        "credit_limit_auto_increase": offer.credit_limit_auto_increase,
        "cooling_off_days": offer.cooling_off_days,
        "recovery_contact_hours": {
            "start": offer.recovery_contact_hours.start,
            "end": offer.recovery_contact_hours.end,
        },
    }

    result = check_loan_compliance(offer_dict, use_rag=False)
    adjusted = auto_adjust_terms(offer_dict, result["violations"])
    adjustments = adjusted.pop("_adjustments_made", [])
    result2 = check_loan_compliance(adjusted, use_rag=False)

    return AutoCorrectionResponse(
        original_violations=result["violation_count"],
        adjusted_offer=adjusted,
        adjustments_made=adjustments,
        post_adjustment_violations=result2["violation_count"],
        post_adjustment_compliant=result2["is_compliant"],
    )


@app.post("/query-guidelines", response_model=list[GuidelineResult], tags=["RAG"])
async def query_guidelines(query: GuidelineQueryInput):
    """Query the RBI Digital Lending Guidelines using semantic search."""
    if not rag_engine or not rag_engine.available:
        raise HTTPException(503, "RAG index not available. Run document_loader.py first.")

    results = rag_engine.query(query.question, query.n_results)
    return [GuidelineResult(**r) for r in results]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8002, reload=True)
