"""
ALIS — FraudSentinel: GraphSAGE Model
=======================================
2-layer GraphSAGE model for binary node classification
(legitimate vs fraud-connected) on UPI transaction networks.

Why GraphSAGE over alternatives:
  • INDUCTIVE — generates embeddings for unseen nodes at inference.
    Every loan applicant is a new node. GCN/GAT are transductive and
    would need full-graph retraining for each new applicant.
  • NEIGHBOURHOOD SAMPLING — scalable to large graphs without
    full-graph message passing (O(k^L * N) vs O(N^2) for full GCN).
  • FRAUD IS RELATIONAL — a synthetic identity looks normal in isolation.
    Only the graph reveals 15 "independent" applicants all transacting
    with the same 3 mule accounts.

Why NOT Isolation Forest alone:
  Isolation Forest operates on tabular node features — it sees each node
  independently. It cannot learn that "this node has 4 neighbours who are
  all connected to 2 common accounts created within 3 days." That's a
  structural pattern, not a feature-space anomaly. GraphSAGE aggregates
  neighbourhood information, making it fundamentally more expressive for
  relational fraud.

Usage:
    python model.py             # train on synthetic graph
    python model.py --epochs 50 # custom epochs
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from graph_builder import (
    NODE_FEATURES,
    build_transaction_graph,
)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── GraphSAGE Model Definition ─────────────────────────────────────────────

class FraudGraphSAGE(torch.nn.Module):
    """
    2-layer GraphSAGE for binary fraud classification.

    Architecture:
        Input (12 features)
        → SAGEConv(12 → 64) + ReLU + Dropout(0.3)
        → SAGEConv(64 → 32) + ReLU + Dropout(0.3)
        → Linear(32 → 2)  (legitimate vs fraud-connected)

    Why 2 layers:
        Each SAGEConv layer aggregates 1-hop neighborhood. With 2 layers,
        each node sees its 2-hop neighborhood — enough to detect the
        fraudster → mule → cashout chain. More layers risk oversmoothing.
    """

    def __init__(self, in_channels: int = 12, hidden_channels: int = 64, out_channels: int = 2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels // 2)
        self.classifier = torch.nn.Linear(hidden_channels // 2, out_channels)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Layer 1: aggregate 1-hop neighborhood
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2: aggregate 2-hop neighborhood
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Classify
        x = self.classifier(x)
        return x

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return the 32-dim node embeddings (pre-classification)."""
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x


# ─── NetworkX → PyG Conversion ──────────────────────────────────────────────

def nx_to_pyg_data(G) -> tuple[Data, list[str]]:
    """
    Convert a NetworkX DiGraph to PyTorch Geometric Data object.

    Returns (data, node_id_list) where node_id_list maps
    PyG integer indices back to original node IDs.
    """
    node_list = list(G.nodes())
    node_to_idx = {nid: i for i, nid in enumerate(node_list)}

    # Build feature matrix
    features = []
    labels = []
    for nid in node_list:
        attrs = G.nodes[nid]
        feat_vec = [float(attrs.get(f, 0.0)) for f in NODE_FEATURES]
        features.append(feat_vec)
        labels.append(int(attrs.get("label", 0)))

    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    # Normalize features (per-feature z-score)
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True) + 1e-8
    x = (x - mean) / std

    # Build edge index
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)

    # Store normalization params for inference
    data.feat_mean = mean
    data.feat_std = std

    return data, node_list


# ─── Training Pipeline ──────────────────────────────────────────────────────

def create_train_val_masks(data: Data, train_ratio: float = 0.7, seed: int = 42):
    """Create stratified train/val masks."""
    rng = np.random.default_rng(seed)
    n = data.num_nodes

    # Stratify: maintain class ratio in train/val
    pos_idx = (data.y == 1).nonzero(as_tuple=True)[0].numpy()
    neg_idx = (data.y == 0).nonzero(as_tuple=True)[0].numpy()

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    pos_split = int(len(pos_idx) * train_ratio)
    neg_split = int(len(neg_idx) * train_ratio)

    train_idx = np.concatenate([pos_idx[:pos_split], neg_idx[:neg_split]])
    val_idx = np.concatenate([pos_idx[pos_split:], neg_idx[neg_split:]])

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    return data


def train_model(
    n_legitimate: int = 80,
    n_mule: int = 15,
    n_fraudster: int = 5,
    epochs: int = 100,
    lr: float = 0.01,
    seed: int = 42,
) -> dict:
    """
    Full training pipeline:
      1. Build synthetic graph
      2. Convert to PyG Data
      3. Train GraphSAGE with class-weighted loss
      4. Evaluate on validation set
      5. Save model + normalization params

    Returns dict of metrics.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Building transaction graph ({n_legitimate}/{n_mule}/{n_fraudster})...")
    G = build_transaction_graph(n_legitimate, n_mule, n_fraudster, seed)

    # Save graph artifacts
    from graph_builder import save_graph
    save_graph(G)

    print("  Converting to PyG format...")
    data, node_list = nx_to_pyg_data(G)
    data = create_train_val_masks(data, train_ratio=0.7, seed=seed)
    data = data.to(DEVICE)

    # Class weights for imbalanced data (legitimate >> fraud)
    n_pos = (data.y == 1).sum().item()
    n_neg = (data.y == 0).sum().item()
    weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float).to(DEVICE)
    print(f"  Class distribution: legitimate={n_neg}, fraud={n_pos}")
    print(f"  Class weights: {weight.tolist()}")

    # Initialize model
    model = FraudGraphSAGE(in_channels=len(NODE_FEATURES)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    print(f"\n  Training GraphSAGE for {epochs} epochs...")
    best_val_f1 = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=weight)
        loss.backward()
        optimizer.step()

        # Validate every 10 epochs
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                val_pred = val_out[data.val_mask].argmax(dim=1).cpu().numpy()
                val_true = data.y[data.val_mask].cpu().numpy()
                val_f1 = f1_score(val_true, val_pred, zero_division=0)
                val_acc = accuracy_score(val_true, val_pred)

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_state = model.state_dict().copy()

                if epoch % 20 == 0 or epoch == epochs:
                    print(f"    Epoch {epoch:3d}  loss={loss.item():.4f}  "
                          f"val_acc={val_acc:.3f}  val_f1={val_f1:.3f}")

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    # ── Final Evaluation ─────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        final_out = model(data.x, data.edge_index)
        final_probs = F.softmax(final_out, dim=1)[:, 1].cpu().numpy()
        final_pred = final_out.argmax(dim=1).cpu().numpy()
        true_labels = data.y.cpu().numpy()

        # Full report
        val_pred = final_out[data.val_mask].argmax(dim=1).cpu().numpy()
        val_true = data.y[data.val_mask].cpu().numpy()
        val_probs = final_probs[data.val_mask.cpu().numpy()]

    report = classification_report(
        val_true, val_pred,
        target_names=["Legitimate", "Fraud-connected"],
        zero_division=0,
    )

    # Compute AUC only if both classes present in val
    try:
        auc = roc_auc_score(val_true, val_probs)
    except ValueError:
        auc = 0.0

    metrics = {
        "val_accuracy": float(accuracy_score(val_true, val_pred)),
        "val_precision": float(precision_score(val_true, val_pred, zero_division=0)),
        "val_recall": float(recall_score(val_true, val_pred, zero_division=0)),
        "val_f1": float(f1_score(val_true, val_pred, zero_division=0)),
        "val_auc_roc": float(auc),
        "total_nodes": int(data.num_nodes),
        "total_edges": int(data.num_edges),
        "fraud_nodes": int(n_pos),
        "epochs_trained": epochs,
    }

    print(f"\n{'='*60}")
    print("  VALIDATION REPORT")
    print(f"{'='*60}")
    print(report)
    print(f"  AUC-ROC: {metrics['val_auc_roc']:.4f}")
    print(f"{'='*60}")

    # ── Save artifacts ───────────────────────────────────────────────────────
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "in_channels": len(NODE_FEATURES),
            "hidden_channels": 64,
            "out_channels": 2,
        },
        "feat_mean": data.feat_mean.cpu(),
        "feat_std": data.feat_std.cpu(),
        "node_features": NODE_FEATURES,
    }, ARTIFACTS_DIR / "fraudsentinel_model.pt")

    # Save node list mapping for scorer
    joblib.dump(node_list, ARTIFACTS_DIR / "node_id_mapping.joblib")

    # Save metrics
    import json
    with open(ARTIFACTS_DIR / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Model saved:  {ARTIFACTS_DIR / 'fraudsentinel_model.pt'}")
    print(f"  Metrics saved: {ARTIFACTS_DIR / 'training_metrics.json'}")

    return metrics


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ALIS FraudSentinel — Train GraphSAGE model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--legit", type=int, default=80)
    parser.add_argument("--mule", type=int, default=15)
    parser.add_argument("--fraud", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  ALIS FraudSentinel — GraphSAGE Training Pipeline")
    print(f"{'='*60}")

    metrics = train_model(
        n_legitimate=args.legit, n_mule=args.mule, n_fraudster=args.fraud,
        epochs=args.epochs, lr=args.lr, seed=args.seed,
    )
    print("\n  ✓ Training complete.\n")


if __name__ == "__main__":
    main()
