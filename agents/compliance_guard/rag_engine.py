"""
ALIS — ComplianceGuard: RAG Query Engine
==========================================
Queries the ChromaDB vector index of RBI Digital Lending Guidelines
to find relevant regulatory clauses for any compliance question.

Used by ComplianceChecker for edge cases that fall outside the
deterministic rule set.

Usage:
    python rag_engine.py "What is the cooling off period?"
"""

import argparse
import sys
from pathlib import Path

from document_loader import CHROMA_DIR, get_collection, COLLECTION_NAME

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


class RBIQueryEngine:
    """
    Semantic search engine over RBI Digital Lending Guidelines.

    Uses ChromaDB with all-MiniLM-L6-v2 embeddings for retrieval.
    No generative LLM in this layer — retrieval only. The compliance
    checker interprets the retrieved clauses.
    """

    def __init__(self, persist_dir: Path = None):
        self.persist_dir = Path(persist_dir or CHROMA_DIR)
        try:
            self.collection = get_collection(self.persist_dir)
            self.available = True
        except Exception as e:
            print(f"  ⚠ RAG index not found: {e}")
            print("  → Run 'python document_loader.py' first.")
            self.available = False

    def query(
        self,
        question: str,
        n_results: int = 5,
    ) -> list[dict]:
        """
        Query the RBI guidelines vector store.

        Parameters
        ----------
        question : str
            Natural language compliance question.
        n_results : int
            Number of relevant chunks to return.

        Returns
        -------
        list of dicts, each with:
            text: str          — the guideline text
            clause: str        — RBI clause reference
            section: str       — section title
            page: int          — page number in the PDF
            relevance_score: float — similarity (lower = more relevant in ChromaDB)
        """
        if not self.available:
            return [{
                "text": "RAG index not available. Run document_loader.py first.",
                "clause": "N/A",
                "section": "N/A",
                "page": 0,
                "relevance_score": 1.0,
            }]

        results = self.collection.query(
            query_texts=[question],
            n_results=min(n_results, self.collection.count()),
        )

        parsed = []
        for i in range(len(results["documents"][0])):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i] if results.get("distances") else 0.0

            parsed.append({
                "text": results["documents"][0][i],
                "clause": meta.get("clause", "N/A"),
                "section": meta.get("section", "N/A"),
                "page": int(meta.get("page", 0)),
                "relevance_score": round(float(distance), 4),
            })

        return parsed

    def query_formatted(self, question: str, n_results: int = 3) -> str:
        """Query and return a formatted string for display / LLM context."""
        results = self.query(question, n_results)

        lines = [f"RBI Guidelines Query: \"{question}\"", "=" * 50]
        for i, r in enumerate(results, 1):
            lines.append(
                f"\n[{i}] Clause {r['clause']} — {r['section']} (p.{r['page']})"
            )
            lines.append(f"    {r['text'][:300]}")
            if len(r['text']) > 300:
                lines.append(f"    ...")

        return "\n".join(lines)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ALIS ComplianceGuard — Query RBI guidelines"
    )
    parser.add_argument(
        "question", nargs="?",
        default="What is the cooling off period for digital loans?",
        help="Question to query",
    )
    parser.add_argument("--n", type=int, default=3, help="Number of results")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  ALIS ComplianceGuard — RAG Query Engine")
    print(f"{'='*60}\n")

    engine = RBIQueryEngine()
    output = engine.query_formatted(args.question, args.n)
    print(output)
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
