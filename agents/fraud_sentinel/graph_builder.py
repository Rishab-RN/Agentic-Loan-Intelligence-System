"""
ALIS — FraudSentinel: Synthetic Transaction Graph Builder
==========================================================
Generates a realistic UPI transaction network graph with three node types:
  • Legitimate users  — organic spending patterns, high reciprocity
  • Mule accounts     — high inflow from fraudsters, rapid cash-out
  • Fraudsters        — star topology to mules, velocity anomalies

The graph structure encodes real UPI fraud typologies observed in India:
  1. Synthetic identity fraud  → new accounts, fake eKYC, rapid activity
  2. Mule account networks     → layering money through intermediaries
  3. Organized fraud rings     → coordinated star/fan-out topologies

Usage:
    python graph_builder.py                      # default: 80/15/5 split
    python graph_builder.py --legit 200 --mule 30 --fraud 10
"""

import argparse
import random
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

# ─── Node Feature Schema ────────────────────────────────────────────────────

NODE_FEATURES = [
    "account_age_days",
    "transaction_count_30d",
    "avg_txn_amount",
    "txn_velocity_per_hour",
    "unique_counterparties",
    "reciprocity_ratio",
    "in_degree",
    "out_degree",
    "degree_ratio",           # in_degree / out_degree
    "max_single_txn_amount",
    "night_txn_ratio",        # txns between 11PM-5AM
    "sim_age_days",
]

NODE_FEATURE_DESCRIPTIONS = {
    "account_age_days": "Days since UPI account activation",
    "transaction_count_30d": "Total UPI transactions in last 30 days",
    "avg_txn_amount": "Average transaction amount (INR)",
    "txn_velocity_per_hour": "Peak transactions per hour (velocity)",
    "unique_counterparties": "Number of unique transaction partners",
    "reciprocity_ratio": "Fraction of counterparties with bidirectional txns",
    "in_degree": "Number of incoming transaction edges",
    "out_degree": "Number of outgoing transaction edges",
    "degree_ratio": "in_degree / out_degree (>1 = net receiver)",
    "max_single_txn_amount": "Largest single transaction amount (INR)",
    "night_txn_ratio": "Fraction of transactions between 11 PM–5 AM",
    "sim_age_days": "Days since SIM card activation",
}


# ─── Node Generators ────────────────────────────────────────────────────────

def _generate_legitimate_node(rng: np.random.Generator, node_id: str) -> dict:
    """Legitimate user: organic patterns, established account, diverse merchants."""
    account_age = rng.integers(90, 1500)
    txn_count = rng.integers(10, 120)
    unique_cp = rng.integers(5, min(txn_count, 40))
    in_deg = rng.integers(3, unique_cp + 1)
    out_deg = rng.integers(3, unique_cp + 1)

    return {
        "node_id": node_id,
        "node_type": "legitimate",
        "label": 0,  # 0 = legitimate
        "account_age_days": int(account_age),
        "transaction_count_30d": int(txn_count),
        "avg_txn_amount": float(rng.uniform(50, 2000)),
        "txn_velocity_per_hour": float(rng.uniform(0.1, 3.0)),
        "unique_counterparties": int(unique_cp),
        "reciprocity_ratio": float(rng.uniform(0.4, 0.9)),
        "in_degree": int(in_deg),
        "out_degree": int(out_deg),
        "degree_ratio": float(in_deg / max(out_deg, 1)),
        "max_single_txn_amount": float(rng.uniform(500, 15000)),
        "night_txn_ratio": float(rng.uniform(0.0, 0.15)),
        "sim_age_days": int(rng.integers(180, 2000)),
    }


def _generate_mule_node(rng: np.random.Generator, node_id: str) -> dict:
    """Mule account: high inflow, rapid cash-out, short tenure, low reciprocity."""
    in_deg = rng.integers(8, 25)
    out_deg = rng.integers(1, 5)

    return {
        "node_id": node_id,
        "node_type": "mule",
        "label": 1,  # 1 = fraud-connected
        "account_age_days": int(rng.integers(5, 60)),
        "transaction_count_30d": int(rng.integers(30, 200)),
        "avg_txn_amount": float(rng.uniform(3000, 25000)),
        "txn_velocity_per_hour": float(rng.uniform(5.0, 30.0)),
        "unique_counterparties": int(rng.integers(2, 8)),
        "reciprocity_ratio": float(rng.uniform(0.0, 0.15)),
        "in_degree": int(in_deg),
        "out_degree": int(out_deg),
        "degree_ratio": float(in_deg / max(out_deg, 1)),
        "max_single_txn_amount": float(rng.uniform(20000, 100000)),
        "night_txn_ratio": float(rng.uniform(0.3, 0.7)),
        "sim_age_days": int(rng.integers(3, 30)),
    }


def _generate_fraudster_node(rng: np.random.Generator, node_id: str) -> dict:
    """Fraudster: synthetic identity, star topology to mules, velocity spikes."""
    out_deg = rng.integers(5, 15)
    in_deg = rng.integers(0, 3)

    return {
        "node_id": node_id,
        "node_type": "fraudster",
        "label": 1,  # 1 = fraud-connected
        "account_age_days": int(rng.integers(1, 20)),
        "transaction_count_30d": int(rng.integers(15, 80)),
        "avg_txn_amount": float(rng.uniform(5000, 50000)),
        "txn_velocity_per_hour": float(rng.uniform(10.0, 50.0)),
        "unique_counterparties": int(rng.integers(3, 10)),
        "reciprocity_ratio": float(rng.uniform(0.0, 0.05)),
        "in_degree": int(in_deg),
        "out_degree": int(out_deg),
        "degree_ratio": float(in_deg / max(out_deg, 1)),
        "max_single_txn_amount": float(rng.uniform(50000, 200000)),
        "night_txn_ratio": float(rng.uniform(0.4, 0.85)),
        "sim_age_days": int(rng.integers(1, 10)),
    }


# ─── Graph Construction ─────────────────────────────────────────────────────

def build_transaction_graph(
    n_legitimate: int = 80,
    n_mule: int = 15,
    n_fraudster: int = 5,
    seed: int = 42,
) -> nx.DiGraph:
    """
    Build a synthetic UPI transaction network with realistic fraud topology.

    Parameters
    ----------
    n_legitimate : int
        Number of legitimate user nodes.
    n_mule : int
        Number of mule account nodes.
    n_fraudster : int
        Number of fraudster nodes.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    nx.DiGraph
        Directed graph with node features and edge transaction data.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)
    G = nx.DiGraph()

    # ── 1. Generate nodes ────────────────────────────────────────────────────
    legit_ids = [f"L_{i:04d}" for i in range(n_legitimate)]
    mule_ids = [f"M_{i:04d}" for i in range(n_mule)]
    fraud_ids = [f"F_{i:04d}" for i in range(n_fraudster)]

    for nid in legit_ids:
        attrs = _generate_legitimate_node(rng, nid)
        G.add_node(nid, **attrs)

    for nid in mule_ids:
        attrs = _generate_mule_node(rng, nid)
        G.add_node(nid, **attrs)

    for nid in fraud_ids:
        attrs = _generate_fraudster_node(rng, nid)
        G.add_node(nid, **attrs)

    # ── 2. Generate edges (transaction relationships) ────────────────────────

    # 2a. Legitimate ↔ Legitimate: organic, reciprocal, diverse
    for nid in legit_ids:
        n_edges = rng.integers(3, min(15, n_legitimate))
        targets = rng.choice(
            [x for x in legit_ids if x != nid],
            size=min(n_edges, len(legit_ids) - 1),
            replace=False,
        )
        for target in targets:
            G.add_edge(nid, target, **_make_legit_edge(rng))
            # ~50% chance of reciprocal edge (organic reciprocity)
            if rng.random() < 0.5:
                G.add_edge(target, nid, **_make_legit_edge(rng))

    # 2b. Legitimate → Merchants (some legitimate users interact loosely)
    # Simulated by connecting a few legit nodes to each other with merchant-like patterns
    merchant_hubs = rng.choice(legit_ids, size=min(5, n_legitimate), replace=False)
    for hub in merchant_hubs:
        customers = rng.choice(
            [x for x in legit_ids if x != hub],
            size=min(rng.integers(5, 20), n_legitimate - 1),
            replace=False,
        )
        for c in customers:
            G.add_edge(c, hub, **_make_legit_edge(rng, merchant=True))

    # 2c. Fraudster → Mule: star topology (the critical fraud pattern)
    for fid in fraud_ids:
        # Each fraudster connects to 3–8 mules
        n_mule_connections = min(rng.integers(3, 9), n_mule)
        connected_mules = rng.choice(mule_ids, size=n_mule_connections, replace=False)
        for mid in connected_mules:
            G.add_edge(fid, mid, **_make_fraud_edge(rng, "fraudster_to_mule"))

    # 2d. Mule → Mule: layering (money moves through mule chain)
    for i, mid in enumerate(mule_ids):
        n_layer = rng.integers(1, 4)
        layer_targets = rng.choice(
            [x for x in mule_ids if x != mid],
            size=min(n_layer, n_mule - 1),
            replace=False,
        )
        for target in layer_targets:
            G.add_edge(mid, target, **_make_fraud_edge(rng, "mule_layering"))

    # 2e. Mule → Legitimate: cash-out (mules withdraw through legit accounts)
    for mid in mule_ids:
        n_cashout = rng.integers(1, 4)
        cashout_targets = rng.choice(legit_ids, size=min(n_cashout, n_legitimate), replace=False)
        for target in cashout_targets:
            G.add_edge(mid, target, **_make_fraud_edge(rng, "mule_cashout"))

    # 2f. A few fraudsters try to blend in with legitimate transactions
    for fid in fraud_ids:
        n_blend = rng.integers(1, 4)
        blend_targets = rng.choice(legit_ids, size=min(n_blend, n_legitimate), replace=False)
        for target in blend_targets:
            G.add_edge(fid, target, **_make_legit_edge(rng))  # camouflage edges

    return G


def _make_legit_edge(rng: np.random.Generator, merchant: bool = False) -> dict:
    """Edge attributes for a legitimate UPI transaction."""
    return {
        "amount": float(rng.uniform(20, 5000) if not merchant else rng.uniform(50, 3000)),
        "txn_type": "P2M" if merchant else "P2P",
        "hour_of_day": int(rng.integers(7, 23)),
        "is_fraud_edge": False,
    }


def _make_fraud_edge(rng: np.random.Generator, edge_type: str) -> dict:
    """Edge attributes for a fraudulent UPI transaction."""
    amounts = {
        "fraudster_to_mule": rng.uniform(10000, 100000),
        "mule_layering": rng.uniform(5000, 50000),
        "mule_cashout": rng.uniform(3000, 30000),
    }
    return {
        "amount": float(amounts.get(edge_type, rng.uniform(5000, 50000))),
        "txn_type": edge_type,
        "hour_of_day": int(rng.integers(0, 6)),  # late night transfers
        "is_fraud_edge": True,
    }


# ─── Export Utilities ────────────────────────────────────────────────────────

def graph_to_dataframes(G: nx.DiGraph) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert the NetworkX graph to node and edge DataFrames for analysis."""
    node_records = []
    for nid, attrs in G.nodes(data=True):
        record = {"node_id": nid}
        record.update(attrs)
        node_records.append(record)

    edge_records = []
    for src, dst, attrs in G.edges(data=True):
        record = {"source": src, "target": dst}
        record.update(attrs)
        edge_records.append(record)

    return pd.DataFrame(node_records), pd.DataFrame(edge_records)


def save_graph(G: nx.DiGraph, output_dir: Path | None = None) -> Path:
    """Save graph as GraphML + node/edge CSVs."""
    output_dir = Path(output_dir or ARTIFACTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # GraphML for NetworkX reload
    nx.write_graphml(G, output_dir / "transaction_graph.graphml")

    # CSVs for inspection
    node_df, edge_df = graph_to_dataframes(G)
    node_df.to_csv(output_dir / "graph_nodes.csv", index=False)
    edge_df.to_csv(output_dir / "graph_edges.csv", index=False)

    return output_dir


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ALIS FraudSentinel — Synthetic UPI transaction graph builder"
    )
    parser.add_argument("--legit", type=int, default=80, help="Legitimate nodes (default: 80)")
    parser.add_argument("--mule", type=int, default=15, help="Mule account nodes (default: 15)")
    parser.add_argument("--fraud", type=int, default=5, help="Fraudster nodes (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  ALIS FraudSentinel — Transaction Graph Builder")
    print(f"{'='*60}")

    G = build_transaction_graph(args.legit, args.mule, args.fraud, args.seed)
    out = save_graph(G)

    n_legit = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "legitimate")
    n_mule = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "mule")
    n_fraud = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "fraudster")
    n_fraud_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("is_fraud_edge"))

    print(f"\n  Nodes:        {G.number_of_nodes()}")
    print(f"    Legitimate: {n_legit}")
    print(f"    Mule:       {n_mule}")
    print(f"    Fraudster:  {n_fraud}")
    print(f"  Edges:        {G.number_of_edges()}")
    print(f"    Fraud edges:{n_fraud_edges}")
    print(f"  Saved to:     {out}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
