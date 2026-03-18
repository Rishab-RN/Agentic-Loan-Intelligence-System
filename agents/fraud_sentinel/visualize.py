"""
ALIS — FraudSentinel: Transaction Graph Visualization
======================================================
Generates publication-ready network visualizations with:
  • Color-coded nodes: green (CLEAN), orange (CAUTION), red (HIGH_RISK/BLOCK)
  • Edge widths proportional to transaction amounts
  • Fraud cluster highlighting

Usage:
    python visualize.py            # generate default visualization
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

from graph_builder import build_transaction_graph

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

# Color palette
COLORS = {
    "legitimate": "#2ecc71",   # green
    "mule": "#e74c3c",         # red
    "fraudster": "#c0392b",    # dark red
    "unknown": "#95a5a6",      # grey
}

RISK_COLORS = {
    "CLEAN": "#2ecc71",
    "CAUTION": "#f39c12",
    "HIGH_RISK": "#e67e22",
    "BLOCK": "#e74c3c",
}


def visualize_transaction_graph(
    G: nx.DiGraph = None,
    save_path: Path = None,
    title: str = "ALIS FraudSentinel — UPI Transaction Network",
    figsize: tuple = (16, 12),
    highlight_node: str = None,
) -> Path:
    """
    Generate a network visualization of the transaction graph.

    Parameters
    ----------
    G : nx.DiGraph
        Transaction graph. If None, builds a default one.
    save_path : Path
        Where to save the PNG.
    title : str
        Plot title.
    figsize : tuple
        Figure dimensions.
    highlight_node : str
        Optional node to highlight with a gold ring.

    Returns
    -------
    Path to saved image.
    """
    if G is None:
        G = build_transaction_graph()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = Path(save_path or ARTIFACTS_DIR / "transaction_graph.png")

    fig, ax = plt.subplots(figsize=figsize, facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # ── Layout: spring with fraud clusters pulled together ───────────────────
    # Use different k values to create visual clustering
    pos = nx.spring_layout(G, k=0.8, iterations=80, seed=42)

    # ── Node colors and sizes ────────────────────────────────────────────────
    node_colors = []
    node_sizes = []
    node_edgecolors = []

    for node_id in G.nodes():
        attrs = G.nodes[node_id]
        node_type = attrs.get("node_type", "unknown")
        node_colors.append(COLORS.get(node_type, COLORS["unknown"]))

        # Size by transaction count
        txn_count = float(attrs.get("transaction_count_30d", 20))
        node_sizes.append(max(60, min(txn_count * 4, 500)))

        if node_id == highlight_node:
            node_edgecolors.append("#ffd700")
        else:
            node_edgecolors.append("#ffffff30")

    # ── Edge styling ─────────────────────────────────────────────────────────
    edge_colors = []
    edge_widths = []
    edge_alphas = []

    for u, v, data in G.edges(data=True):
        is_fraud = data.get("is_fraud_edge", False)
        amount = float(data.get("amount", 100))

        if is_fraud:
            edge_colors.append("#e74c3c")
            edge_widths.append(min(amount / 10000, 3.0))
            edge_alphas.append(0.6)
        else:
            edge_colors.append("#3498db40")
            edge_widths.append(min(amount / 5000, 1.5))
            edge_alphas.append(0.2)

    # ── Draw edges ───────────────────────────────────────────────────────────
    # Draw legitimate edges first (behind)
    legit_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get("is_fraud_edge")]
    fraud_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("is_fraud_edge")]

    nx.draw_networkx_edges(
        G, pos, edgelist=legit_edges, ax=ax,
        edge_color="#3498db", alpha=0.12, width=0.5,
        arrows=True, arrowsize=6, arrowstyle="-|>",
        connectionstyle="arc3,rad=0.1",
    )

    nx.draw_networkx_edges(
        G, pos, edgelist=fraud_edges, ax=ax,
        edge_color="#e74c3c", alpha=0.5, width=1.5,
        arrows=True, arrowsize=10, arrowstyle="-|>",
        connectionstyle="arc3,rad=0.15",
    )

    # ── Draw nodes ───────────────────────────────────────────────────────────
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors=node_edgecolors,
        linewidths=[3 if nid == highlight_node else 0.5 for nid in G.nodes()],
        alpha=0.85,
    )

    # ── Labels for fraud nodes ───────────────────────────────────────────────
    fraud_labels = {
        nid: nid for nid in G.nodes()
        if G.nodes[nid].get("node_type") in ("mule", "fraudster")
    }
    if highlight_node:
        fraud_labels[highlight_node] = highlight_node

    nx.draw_networkx_labels(
        G, pos, labels=fraud_labels, ax=ax,
        font_size=6, font_color="white", font_weight="bold",
    )

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=COLORS["legitimate"], label=f"Legitimate ({sum(1 for _, d in G.nodes(data=True) if d.get('node_type') == 'legitimate')})"),
        mpatches.Patch(color=COLORS["mule"], label=f"Mule Account ({sum(1 for _, d in G.nodes(data=True) if d.get('node_type') == 'mule')})"),
        mpatches.Patch(color=COLORS["fraudster"], label=f"Fraudster ({sum(1 for _, d in G.nodes(data=True) if d.get('node_type') == 'fraudster')})"),
        mpatches.Patch(color="#3498db", label="Legitimate Txn", alpha=0.3),
        mpatches.Patch(color="#e74c3c", label="Fraud Txn", alpha=0.7),
    ]

    ax.legend(
        handles=legend_patches, loc="upper left",
        fontsize=10, facecolor="#16213e", edgecolor="#e94560",
        labelcolor="white", framealpha=0.9,
    )

    # ── Title and stats ──────────────────────────────────────────────────────
    ax.set_title(title, fontsize=16, color="white", pad=20, fontweight="bold")

    stats_text = (
        f"Nodes: {G.number_of_nodes()} | "
        f"Edges: {G.number_of_edges()} | "
        f"Fraud edges: {sum(1 for _, _, d in G.edges(data=True) if d.get('is_fraud_edge'))}"
    )
    ax.text(
        0.5, -0.02, stats_text,
        transform=ax.transAxes, ha="center",
        fontsize=10, color="#a0a0a0",
    )

    ax.axis("off")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)

    print(f"  Graph visualization saved: {save_path}")
    return save_path


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ALIS FraudSentinel — Graph visualization")
    parser.add_argument("--highlight", type=str, default=None, help="Node ID to highlight")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  ALIS FraudSentinel — Transaction Graph Visualization")
    print(f"{'='*60}\n")

    # Try loading existing graph
    graphml = ARTIFACTS_DIR / "transaction_graph.graphml"
    if graphml.exists():
        print("  Loading saved graph...")
        G = nx.read_graphml(graphml)
        # Restore types
        for nid in G.nodes():
            for key in G.nodes[nid]:
                try:
                    G.nodes[nid][key] = float(G.nodes[nid][key])
                except (ValueError, TypeError):
                    pass
        for u, v in G.edges():
            edge = G.edges[u, v]
            if "is_fraud_edge" in edge:
                edge["is_fraud_edge"] = str(edge["is_fraud_edge"]).lower() in ("true", "1")
            if "amount" in edge:
                edge["amount"] = float(edge["amount"])
    else:
        print("  Building new graph...")
        G = build_transaction_graph()

    visualize_transaction_graph(G, highlight_node=args.highlight)
    print(f"\n  ✓ Done.\n")


if __name__ == "__main__":
    main()
