"""
ALIS — ComplianceGuard: Document Loader & Vector Indexer
=========================================================
Loads the RBI Digital Lending Guidelines PDF, chunks it into
300-token segments with 50-token overlap, and indexes into a
ChromaDB vector store using all-MiniLM-L6-v2 embeddings.

The embedding model is 80MB, runs on CPU, and achieves 0.78 nDCG
on MTEB retrieval — precise enough for a 40-page regulatory document.

If no PDF is available, the module falls back to a built-in
representation of the key RBI guidelines for demo/competition use.

Usage:
    python document_loader.py                     # index built-in guidelines
    python document_loader.py --pdf path/to/rbi.pdf  # index actual PDF
"""

import argparse
import hashlib
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
CHROMA_DIR = ARTIFACTS_DIR / "chroma_db"
COLLECTION_NAME = "rbi_digital_lending_guidelines"

# Embedding model: all-MiniLM-L6-v2 (80MB, CPU, free)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking config
CHUNK_SIZE_TOKENS = 300
CHUNK_OVERLAP_TOKENS = 50

# ─── Built-In RBI Guidelines (for demo when PDF unavailable) ────────────────

RBI_GUIDELINES_BUILTIN = [
    {
        "clause": "1.1",
        "section": "Scope and Applicability",
        "text": (
            "These guidelines are applicable to all digital lending activities "
            "by Regulated Entities (REs) as well as Lending Service Providers (LSPs) "
            "and Digital Lending Apps (DLAs) engaged by the REs for extending any "
            "credit facility. Digital lending refers to a remote and automated lending "
            "process involving minimal human interaction, leveraging technology, "
            "analytics, and algorithms to process loan origination, disbursement, "
            "and recovery."
        ),
        "page": 2,
    },
    {
        "clause": "2.1",
        "section": "Key Fact Statement (KFS)",
        "text": (
            "A standardised Key Fact Statement (KFS) must be provided to the borrower "
            "before the execution of the loan contract. The KFS shall include the "
            "Annual Percentage Rate (APR), details of all fees and charges, including "
            "penal charges and the annualised rate of recovery, the tenure of the loan, "
            "the cooling-off/look-up period during which the borrower shall not be "
            "charged any penalty on prepayment of digital loan."
        ),
        "page": 4,
    },
    {
        "clause": "2.2",
        "section": "Annual Percentage Rate (APR)",
        "text": (
            "All-inclusive cost of digital loans shall be disclosed upfront in the form "
            "of Annual Percentage Rate (APR) to the borrower. The APR shall include "
            "the processing fee, insurance charges, documentation charges, and any "
            "other fees or charges payable by the borrower to the RE or the LSP. "
            "The computation of APR shall be on a standardised basis as prescribed "
            "by the RBI."
        ),
        "page": 5,
    },
    {
        "clause": "2.3",
        "section": "Cooling-Off/Look-Up Period",
        "text": (
            "A cooling-off/look-up period shall be provided to the borrower to exit "
            "the digital loan by paying the principal and the proportionate APR. "
            "The cooling-off period shall not be less than three (3) days for loans "
            "having a tenure of seven days or more. During the cooling-off period, "
            "no penalty shall be charged to the borrower on prepayment or foreclosure "
            "of the loan. The borrower shall also be entitled to a complete refund of "
            "any fees charged during the look-up period."
        ),
        "page": 6,
    },
    {
        "clause": "3.1",
        "section": "Loan Disbursement",
        "text": (
            "All loan disbursals shall be made only into the bank account of the "
            "borrower. Disbursals cannot be made to a third-party account or "
            "prepaid instrument. The disbursal and repayments should be executed "
            "directly between the bank account of the borrower and the RE, without "
            "any pass-through or pool account of the Lending Service Provider. "
            "This ensures full transparency and traceability of fund flows."
        ),
        "page": 8,
    },
    {
        "clause": "3.2",
        "section": "KYC and Due Diligence",
        "text": (
            "Know Your Customer (KYC) verification must be completed before loan "
            "disbursement. The RE shall ensure that the borrower's identity is "
            "verified through Aadhaar-based eKYC, video KYC, or any other method "
            "prescribed by the RBI. No loan shall be disbursed without the completion "
            "of KYC. Additionally, the RE must maintain records of KYC documents "
            "in accordance with the Prevention of Money Laundering Act (PMLA)."
        ),
        "page": 9,
    },
    {
        "clause": "4.1",
        "section": "Data Collection and Privacy",
        "text": (
            "Any collection of data by DLAs should be need-based and with prior "
            "and explicit consent of the borrower. The DLAs shall have a comprehensive "
            "privacy policy detailing the types of data collected, purpose, storage, "
            "usage and sharing with third parties. DLAs shall collect only the data "
            "that is necessary for the credit assessment or loan servicing. Access to "
            "phone resources such as contact list, files, media, call logs, telephony "
            "functions shall not be permitted."
        ),
        "page": 11,
    },
    {
        "clause": "4.2",
        "section": "Data Storage",
        "text": (
            "All personal data of borrowers stored by DLAs shall be stored in servers "
            "located within India. The data shall not be transferred to any third party "
            "without the explicit consent of the borrower. Data retention policy shall "
            "be clearly disclosed and data shall be deleted upon borrower request or "
            "after a reasonable period following loan closure. Borrower data must be "
            "encrypted at rest and in transit."
        ),
        "page": 12,
    },
    {
        "clause": "5.1",
        "section": "Credit Limit Management",
        "text": (
            "No automatic increase in the credit limit shall be permitted without the "
            "explicit consent of the borrower. The RE must communicate any proposed "
            "increase in the credit limit to the borrower and obtain explicit acceptance "
            "before it takes effect. This includes credit lines, overdraft facilities, "
            "and Buy Now Pay Later (BNPL) products."
        ),
        "page": 14,
    },
    {
        "clause": "5.2",
        "section": "Fair Lending Practices",
        "text": (
            "The Annual Percentage Rate (APR) charged by digital lenders must be "
            "reasonable and not usurious. While no specific cap has been mandated, "
            "APRs exceeding 36% per annum for unsecured microloans shall require "
            "additional justification documented in the credit file. APRs exceeding "
            "50% per annum are considered prima facie exploitative and may attract "
            "regulatory action."
        ),
        "page": 15,
    },
    {
        "clause": "6.1",
        "section": "Recovery Practices",
        "text": (
            "Recovery agents appointed by REs must adhere to the Fair Practices Code. "
            "Agents shall not resort to intimidation, physical threats, or any form of "
            "harassment. Recovery-related communication with the borrower shall only be "
            "made during reasonable hours between 8:00 AM and 8:00 PM. No recovery "
            "agent shall contact the borrower's family members, employers, or references "
            "without prior consent. The borrower shall be informed of the name and "
            "details of the recovery agent before any contact is made."
        ),
        "page": 17,
    },
    {
        "clause": "6.2",
        "section": "Penal Charges",
        "text": (
            "Penal charges for delayed payments must be reasonable, clearly disclosed "
            "upfront, and shall not be compounded. The penal interest rate must be "
            "disclosed in the KFS and loan agreement. Penal charges shall be levied "
            "on the outstanding loan amount only and not on the entire loan amount."
        ),
        "page": 18,
    },
    {
        "clause": "7.1",
        "section": "Grievance Redressal",
        "text": (
            "The RE shall have a three-tier grievance redressal mechanism: (1) the "
            "RE's own customer care, (2) the nodal grievance redressal officer designated "
            "by the RE, and (3) the RBI's Integrated Ombudsman mechanism. The details "
            "of the grievance redressal officer must be displayed on the DLA and the "
            "website of the RE. Complaints must be resolved within 30 days."
        ),
        "page": 20,
    },
    {
        "clause": "8.1",
        "section": "Reporting and Transparency",
        "text": (
            "All digital lending products shall report loan details to Credit "
            "Information Companies (CICs) recognized by the RBI. The RE must ensure "
            "that borrower credit behaviour is accurately reported. Additionally, "
            "the RE must disclose the name of the LSP on the loan agreement and "
            "clearly state that the lending is being done by the RE and not by the LSP."
        ),
        "page": 22,
    },
]


# ─── PDF Loading ─────────────────────────────────────────────────────────────

def load_pdf_chunks(pdf_path: str) -> list[dict]:
    """
    Load an RBI guidelines PDF and chunk into overlapping segments.

    Uses word-level tokenization for chunking (approximating tokens
    as whitespace-separated words — close enough for embedding models).
    """
    from PyPDF2 import PdfReader

    reader = PdfReader(pdf_path)
    chunks = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text:
            continue

        words = text.split()
        start = 0
        while start < len(words):
            end = min(start + CHUNK_SIZE_TOKENS, len(words))
            chunk_text = " ".join(words[start:end])

            if len(chunk_text.strip()) > 20:
                chunks.append({
                    "text": chunk_text,
                    "page": page_num,
                    "chunk_id": f"pdf_p{page_num}_{start}",
                    "source": "rbi_digital_lending_guidelines_pdf",
                })

            start += CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS

    return chunks


def load_builtin_chunks() -> list[dict]:
    """Convert built-in guideline sections into chunks."""
    chunks = []
    for guideline in RBI_GUIDELINES_BUILTIN:
        chunks.append({
            "text": (
                f"[Clause {guideline['clause']}] {guideline['section']}: "
                f"{guideline['text']}"
            ),
            "page": guideline["page"],
            "clause": guideline["clause"],
            "section": guideline["section"],
            "chunk_id": f"builtin_{guideline['clause']}",
            "source": "rbi_builtin_guidelines",
        })
    return chunks


# ─── Vector Index ────────────────────────────────────────────────────────────

def build_vector_index(
    chunks: list[dict],
    persist_dir: Path = None,
) -> chromadb.Collection:
    """
    Build a ChromaDB vector index from document chunks.

    Uses all-MiniLM-L6-v2 for embeddings (80MB, CPU, free, 384-dim).
    """
    persist_dir = Path(persist_dir or CHROMA_DIR)
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Use sentence-transformers embedding function
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
    )

    client = chromadb.PersistentClient(path=str(persist_dir))

    # Delete existing collection if any, recreate fresh
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"description": "RBI Digital Lending Guidelines 2022/2023"},
    )

    # Add chunks to collection
    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{k: v for k, v in c.items() if k != "text"} for c in chunks]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    return collection


def get_collection(persist_dir: Path = None) -> chromadb.Collection:
    """Load existing ChromaDB collection."""
    persist_dir = Path(persist_dir or CHROMA_DIR)

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
    )

    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
    )


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ALIS ComplianceGuard — Index RBI Digital Lending Guidelines"
    )
    parser.add_argument(
        "--pdf", type=str, default=None,
        help="Path to RBI Digital Lending Guidelines PDF",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  ALIS ComplianceGuard — Document Indexer")
    print(f"{'='*60}\n")

    if args.pdf and Path(args.pdf).exists():
        print(f"  Loading PDF: {args.pdf}")
        chunks = load_pdf_chunks(args.pdf)
        print(f"  Extracted {len(chunks)} chunks from PDF")
    else:
        if args.pdf:
            print(f"  ⚠ PDF not found: {args.pdf}")
        print("  Using built-in RBI guidelines (14 key clauses)")
        chunks = load_builtin_chunks()

    print(f"  Building vector index with {EMBEDDING_MODEL}...")
    collection = build_vector_index(chunks)

    print(f"\n  ✓ Index built: {collection.count()} documents")
    print(f"  ✓ Persisted to: {CHROMA_DIR}")

    # Quick test query
    print(f"\n  Testing query: 'cooling off period for digital loans'")
    results = collection.query(
        query_texts=["cooling off period for digital loans"],
        n_results=2,
    )
    for i, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0])
    ):
        clause = meta.get("clause", "N/A")
        print(f"    [{i+1}] Clause {clause}: {doc[:120]}...")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
