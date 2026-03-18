"""
ALIS — FraudSentinel: Fraud Risk Scorer
========================================
Scores individual loan applicants for fraud risk using the trained
GraphSAGE model + graph-structural heuristics.

Returns:
  • fraud_risk_score (0-100)
  • risk_level: CLEAN / CAUTION / HIGH_RISK / BLOCK
  • explanation: which features triggered the flag
  • connected_suspicious_accounts: list of flagged neighbors

Usage:
    python scorer.py                    # score a sample applicant
    python scorer.py --node L_0010      # score a specific node
"""

import argparse
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

from graph_builder import NODE_FEATURES, NODE_FEATURE_DESCRIPTIONS, build_transaction_graph
from model import DEVICE, FraudGraphSAGE, nx_to_pyg_data

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

# Risk thresholds
THRESHOLDS = {
    "CLEAN": (0, 25),
    "CAUTION": (25, 50),
    "HIGH_RISK": (50, 75),
    "BLOCK": (75, 100),
}


class FraudScorer:
    """
    Production fraud scorer combining GraphSAGE predictions with
    graph-structural heuristics for robust fraud assessment.
    """

    def __init__(self, model_path=None, graph=None):
        model_path = Path(model_path or ARTIFACTS_DIR / "fraudsentinel_model.pt")

        # Load trained model
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        config = checkpoint["model_config"]
        self.model = FraudGraphSAGE(**config).to(DEVICE)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.feat_mean = checkpoint["feat_mean"].to(DEVICE)
        self.feat_std = checkpoint["feat_std"].to(DEVICE)

        # Load or build graph
        if graph is not None:
            self.G = graph
        else:
            graphml = ARTIFACTS_DIR / "transaction_graph.graphml"
            if graphml.exists():
                self.G = nx.read_graphml(graphml)
                # Restore numeric types from GraphML string storage
                for nid in self.G.nodes():
                    for feat in NODE_FEATURES:
                        if feat in self.G.nodes[nid]:
                            self.G.nodes[nid][feat] = float(self.G.nodes[nid][feat])
                    if "label" in self.G.nodes[nid]:
                        self.G.nodes[nid]["label"] = int(self.G.nodes[nid]["label"])
            else:
                self.G = build_transaction_graph()

        # Convert to PyG for inference
        self.pyg_data, self.node_list = nx_to_pyg_data(self.G)
        self.node_to_idx = {nid: i for i, nid in enumerate(self.node_list)}
        self.pyg_data = self.pyg_data.to(DEVICE)

        # Pre-compute all fraud probabilities
        with torch.no_grad():
            out = self.model(self.pyg_data.x, self.pyg_data.edge_index)
            self.all_probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()

    def score_applicant(self, applicant_node_id: str) -> dict:
        """
        Score a single applicant for fraud risk.

        Parameters
        ----------
        applicant_node_id : str
            Node ID in the transaction graph.

        Returns
        -------
        dict with:
            fraud_risk_score: int (0-100)
            risk_level: str (CLEAN/CAUTION/HIGH_RISK/BLOCK)
            gnn_probability: float (raw GraphSAGE fraud probability)
            explanation: list of str (human-readable risk factors)
            connected_suspicious_accounts: list of dict
            node_features: dict (raw feature values)
        """
        if applicant_node_id not in self.node_to_idx:
            return {
                "fraud_risk_score": 50,
                "risk_level": "CAUTION",
                "gnn_probability": None,
                "explanation": ["Node not found in transaction graph — cannot verify network."],
                "connected_suspicious_accounts": [],
                "node_features": {},
            }

        idx = self.node_to_idx[applicant_node_id]
        node_attrs = dict(self.G.nodes[applicant_node_id])

        # ── 1. GNN-based fraud probability ───────────────────────────────────
        gnn_prob = float(self.all_probs[idx])

        # ── 2. Structural heuristics (supplement GNN) ────────────────────────
        heuristic_score, explanations = self._compute_heuristics(applicant_node_id, node_attrs)

        # ── 3. Combined score: 70% GNN + 30% heuristics ─────────────────────
        combined = 0.7 * (gnn_prob * 100) + 0.3 * heuristic_score
        fraud_risk_score = int(np.clip(combined, 0, 100))

        # ── 4. Risk level ────────────────────────────────────────────────────
        risk_level = "CLEAN"
        for level, (low, high) in THRESHOLDS.items():
            if low <= fraud_risk_score < high:
                risk_level = level
                break
        if fraud_risk_score >= 75:
            risk_level = "BLOCK"

        # ── 5. Connected suspicious accounts ─────────────────────────────────
        suspicious = self._find_suspicious_neighbors(applicant_node_id)

        # ── 6. Compile features for audit ────────────────────────────────────
        features = {f: node_attrs.get(f, 0.0) for f in NODE_FEATURES}

        return {
            "fraud_risk_score": fraud_risk_score,
            "risk_level": risk_level,
            "gnn_probability": round(gnn_prob, 4),
            "explanation": explanations,
            "connected_suspicious_accounts": suspicious,
            "node_features": features,
        }

    def _compute_heuristics(self, node_id: str, attrs: dict) -> tuple[float, list[str]]:
        """Compute rule-based heuristic fraud score + explanations."""
        score = 0.0
        explanations = []

        # H1: Account age (new accounts are suspicious)
        age = float(attrs.get("account_age_days", 365))
        if age < 15:
            score += 30
            explanations.append(f"🔴 Very new account ({age:.0f} days) — synthetic identity risk")
        elif age < 45:
            score += 15
            explanations.append(f"🟡 New account ({age:.0f} days) — limited history")

        # H2: Transaction velocity (spikes indicate automated fraud)
        velocity = float(attrs.get("txn_velocity_per_hour", 0))
        if velocity > 15:
            score += 25
            explanations.append(f"🔴 Extreme txn velocity ({velocity:.1f}/hr) — automated pattern")
        elif velocity > 8:
            score += 12
            explanations.append(f"🟡 High txn velocity ({velocity:.1f}/hr) — above normal")

        # H3: Reciprocity (legitimate users have bidirectional relationships)
        reciprocity = float(attrs.get("reciprocity_ratio", 0.5))
        if reciprocity < 0.05:
            score += 20
            explanations.append(f"🔴 Near-zero reciprocity ({reciprocity:.2f}) — one-way money flow")
        elif reciprocity < 0.2:
            score += 8
            explanations.append(f"🟡 Low reciprocity ({reciprocity:.2f}) — limited mutual transactions")

        # H4: Night transaction ratio
        night_ratio = float(attrs.get("night_txn_ratio", 0))
        if night_ratio > 0.5:
            score += 15
            explanations.append(f"🔴 Unusual night activity ({night_ratio:.0%} between 11PM-5AM)")
        elif night_ratio > 0.3:
            score += 5
            explanations.append(f"🟡 Elevated night activity ({night_ratio:.0%} between 11PM-5AM)")

        # H5: SIM age vs account age mismatch
        sim_age = float(attrs.get("sim_age_days", 365))
        if sim_age < 10:
            score += 15
            explanations.append(f"🔴 Brand new SIM ({sim_age:.0f} days) — possible burner device")

        # H6: Degree ratio (high in-degree, low out-degree = mule pattern)
        degree_ratio = float(attrs.get("degree_ratio", 1.0))
        if degree_ratio > 5:
            score += 15
            explanations.append(f"🔴 Extreme inflow bias (ratio {degree_ratio:.1f}) — mule account pattern")

        # Clamp to 0-100
        score = float(np.clip(score, 0, 100))

        if not explanations:
            explanations.append("✅ No structural red flags detected in transaction pattern")

        return score, explanations

    def _find_suspicious_neighbors(self, node_id: str) -> list[dict]:
        """Find neighbors with high fraud scores."""
        suspicious = []

        # Check all neighbors (both predecessors and successors)
        neighbors = set(self.G.predecessors(node_id)) | set(self.G.successors(node_id))

        for neighbor_id in neighbors:
            if neighbor_id not in self.node_to_idx:
                continue

            n_idx = self.node_to_idx[neighbor_id]
            n_prob = float(self.all_probs[n_idx])
            n_attrs = self.G.nodes[neighbor_id]

            if n_prob > 0.4:
                # Get edge info
                edge_data = self.G.get_edge_data(node_id, neighbor_id) or \
                            self.G.get_edge_data(neighbor_id, node_id) or {}

                suspicious.append({
                    "account_id": neighbor_id,
                    "node_type": n_attrs.get("node_type", "unknown"),
                    "fraud_probability": round(n_prob, 3),
                    "account_age_days": int(float(n_attrs.get("account_age_days", 0))),
                    "connection_type": edge_data.get("txn_type", "unknown"),
                    "txn_amount": float(edge_data.get("amount", 0)),
                })

        # Sort by fraud probability descending
        suspicious.sort(key=lambda x: x["fraud_probability"], reverse=True)
        return suspicious[:10]  # cap at 10


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ALIS FraudSentinel — Fraud risk scorer")
    parser.add_argument("--node", type=str, default=None, help="Node ID to score")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  ALIS FraudSentinel — Fraud Risk Scorer")
    print(f"{'='*60}\n")

    scorer = FraudScorer()

    # Score sample nodes from each category
    test_nodes = [args.node] if args.node else ["L_0010", "M_0005", "F_0002"]

    for node_id in test_nodes:
        result = scorer.score_applicant(node_id)

        level_emoji = {
            "CLEAN": "✅", "CAUTION": "⚠️",
            "HIGH_RISK": "🔶", "BLOCK": "🚫",
        }

        print(f"  ── {node_id} {'─'*45}")
        print(f"  Risk Score:  {result['fraud_risk_score']}/100")
        print(f"  Risk Level:  {level_emoji.get(result['risk_level'], '')} {result['risk_level']}")
        print(f"  GNN Prob:    {result['gnn_probability']}")
        print(f"  Explanations:")
        for exp in result["explanation"]:
            print(f"    {exp}")

        if result["connected_suspicious_accounts"]:
            print(f"  Suspicious Connections ({len(result['connected_suspicious_accounts'])}):")
            for acc in result["connected_suspicious_accounts"][:5]:
                print(f"    {acc['account_id']} [{acc['node_type']}]  "
                      f"fraud_prob={acc['fraud_probability']:.3f}  "
                      f"age={acc['account_age_days']}d")
        print()

    print(f"  ✓ Scoring complete.\n")


if __name__ == "__main__":
    main()
