"""
ALIS — ComplianceGuard: Audit Trail Logger
============================================
Logs every compliance check to a SQLite database for regulatory audit.

Why SQLite (not PostgreSQL for the MVP):
  - Zero setup — no server, no Docker, no credentials
  - SQLite handles 100K+ writes/day — enough for any NBFC demo
  - Trivially portable — one .db file to show the judges
  - Migration to PostgreSQL is a 10-line SQLAlchemy adapter change

Schema mirrors what an RBI auditor would want:
  - Who was checked, when, what was the offer, what violations were found,
    what was the decision, and a hash for tamper detection.

Usage:
    from audit_logger import AuditLogger
    logger = AuditLogger()
    logger.log_check(applicant_id, loan_offer, result)
"""

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
DB_PATH = ARTIFACTS_DIR / "compliance_audit.db"


class AuditLogger:
    """
    SQLite-backed audit trail for compliance checks.

    Every row is append-only with a SHA-256 chain hash for tamper detection.
    """

    def __init__(self, db_path: Path = None):
        self.db_path = Path(db_path or DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create the audit table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS compliance_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                applicant_id TEXT NOT NULL,
                offer_hash TEXT NOT NULL,
                loan_offer_json TEXT NOT NULL,
                is_compliant INTEGER NOT NULL,
                violation_count INTEGER NOT NULL,
                violations_json TEXT NOT NULL,
                corrections_json TEXT NOT NULL,
                decision TEXT NOT NULL,
                chain_hash TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_applicant
            ON compliance_audit(applicant_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON compliance_audit(timestamp)
        """)
        conn.commit()
        conn.close()

    def _get_last_hash(self, conn: sqlite3.Connection) -> str:
        """Get the chain_hash of the last entry for hash chain."""
        row = conn.execute(
            "SELECT chain_hash FROM compliance_audit ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else "GENESIS"

    def log_check(
        self,
        applicant_id: str,
        loan_offer: dict,
        check_result: dict,
    ) -> int:
        """
        Log a compliance check to the audit trail.

        Parameters
        ----------
        applicant_id : str
            Applicant identifier.
        loan_offer : dict
            The loan offer that was checked.
        check_result : dict
            Output from check_loan_compliance().

        Returns
        -------
        int: Row ID of the inserted audit entry.
        """
        conn = sqlite3.connect(self.db_path)

        timestamp = datetime.utcnow().isoformat()
        offer_json = json.dumps(loan_offer, sort_keys=True, default=str)
        violations_json = json.dumps(check_result.get("violations", []), default=str)
        corrections_json = json.dumps(
            check_result.get("recommended_corrections", []), default=str
        )

        # Decision logic
        if check_result.get("is_compliant"):
            decision = "PASS"
        elif check_result.get("has_critical_violations"):
            decision = "HARD_BLOCK"
        else:
            decision = "NEEDS_CORRECTION"

        # Chain hash for tamper detection
        last_hash = self._get_last_hash(conn)
        chain_input = f"{last_hash}|{timestamp}|{applicant_id}|{offer_json}"
        chain_hash = hashlib.sha256(chain_input.encode()).hexdigest()[:32]

        cursor = conn.execute(
            """
            INSERT INTO compliance_audit
            (timestamp, applicant_id, offer_hash, loan_offer_json,
             is_compliant, violation_count, violations_json,
             corrections_json, decision, chain_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                applicant_id,
                check_result.get("offer_hash", ""),
                offer_json,
                int(check_result.get("is_compliant", False)),
                check_result.get("violation_count", 0),
                violations_json,
                corrections_json,
                decision,
                chain_hash,
            ),
        )
        row_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return row_id

    def get_audit_trail(
        self,
        applicant_id: str = None,
        limit: int = 50,
    ) -> list[dict]:
        """Retrieve audit entries, optionally filtered by applicant."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        if applicant_id:
            rows = conn.execute(
                "SELECT * FROM compliance_audit WHERE applicant_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (applicant_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM compliance_audit ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()

        conn.close()
        return [dict(row) for row in rows]

    def get_stats(self) -> dict:
        """Get aggregate audit statistics."""
        conn = sqlite3.connect(self.db_path)

        total = conn.execute("SELECT COUNT(*) FROM compliance_audit").fetchone()[0]
        compliant = conn.execute(
            "SELECT COUNT(*) FROM compliance_audit WHERE is_compliant = 1"
        ).fetchone()[0]
        blocked = conn.execute(
            "SELECT COUNT(*) FROM compliance_audit WHERE decision = 'HARD_BLOCK'"
        ).fetchone()[0]

        conn.close()

        return {
            "total_checks": total,
            "compliant": compliant,
            "non_compliant": total - compliant,
            "hard_blocked": blocked,
            "compliance_rate": round(compliant / max(total, 1), 4),
        }

    def verify_chain(self) -> bool:
        """Verify the hash chain integrity (tamper detection)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM compliance_audit ORDER BY id ASC"
        ).fetchall()
        conn.close()

        prev_hash = "GENESIS"
        for row in rows:
            chain_input = (
                f"{prev_hash}|{row['timestamp']}|{row['applicant_id']}|"
                f"{row['loan_offer_json']}"
            )
            expected = hashlib.sha256(chain_input.encode()).hexdigest()[:32]
            if row["chain_hash"] != expected:
                return False
            prev_hash = row["chain_hash"]

        return True
