"""
training.py
-----------
Training loop, metric helpers, and entry point for the GCVA pipeline.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve
from torch_geometric.loader import DataLoader as GeoDataLoader

from .dataset import ThreeViewsDataset, collate_three
from .model import GCVA, gcva_loss
from .plots import save_conf_roc, save_training_curves, plot_fusion_weights_violin, plot_fusion_weights_ternary


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def read_yaml(path: str) -> dict:
    """Load a YAML configuration file."""
    import yaml
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def require_keys(d: dict, keys: list, root: str = "cfg") -> None:
    """Raise KeyError if any *key* is missing from *d*."""
    for k in keys:
        if k not in d:
            raise KeyError(f"Missing key '{k}' in {root}")


def set_deterministic_seed(seed: int) -> None:
    """Set all random seeds and enable deterministic PyTorch operations."""
    import random
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def compute_pos_weight_from_labels(y_np: np.ndarray) -> float:
    """Compute BCEWithLogitsLoss positive weight from label counts."""
    y = np.asarray(y_np, dtype=int)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return 1.0 if pos == 0 else float(neg / max(1, pos))


def best_threshold_from_probs(probs: np.ndarray, labels: np.ndarray) -> float:
    """Select the decision threshold that maximises F1 on *probs*."""
    pr, rc, th = precision_recall_curve(labels, probs)
    f1 = 2 * (pr * rc) / (pr + rc + 1e-6)
    if f1.size == 0:
        return 0.5
    idx = int(np.argmax(f1))
    return 0.5 if idx >= len(th) else float(th[idx])


def f1_at_threshold(probs, labels, thr: float) -> float:
    """Compute F1 score at a fixed decision threshold."""
    p = (np.asarray(probs) > thr).astype(int)
    y = np.asarray(labels).astype(int)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    prec = tp / max(1, tp + fp)
    rec  = tp / max(1, tp + fn)
    return float(2 * prec * rec / (prec + rec + 1e-6))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 80)
    print("STARTING GCVA TRAINING")
    print("=" * 80 + "\n")

    print("[1/10] Loading configuration...")
    cfg = read_yaml("configs/config_three_gcva.yaml")
    if "tabular_features" not in cfg and "tab_features" in cfg:
        cfg["tabular_features"] = cfg.pop("tab_features")
    require_keys(cfg, [
        "seed", "csv_paths", "target", "tabular_features",
        "gnn_model_params", "tab_model_params", "temporal_params",
        "fusion_params", "training_params", "output_dir",
    ], "config")
    require_keys(cfg["csv_paths"], ["graphs", "tabular", "temporal"], "csv_paths")
    require_keys(cfg["gnn_model_params"], ["hidden_dim"], "gnn_model_params")
    require_keys(cfg["tab_model_params"], ["hidden_dim"], "tab_model_params")
    require_keys(cfg["temporal_params"], ["window"], "temporal_params")
    require_keys(cfg["training_params"], ["epochs", "learning_rate", "batch_size", "patience"], "training_params")
    print("   ✓ Configuration loaded")

    seed = int(cfg["seed"])
    set_deterministic_seed(seed)
    ensure_dir(cfg["output_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   ✓ Device: {device} | Seed: {seed}")

    print("\n[2/10] Loading datasets...")
    ds = ThreeViewsDataset(
        graphs_csv=cfg["csv_paths"]["graphs"],
        tabular_csv=cfg["csv_paths"]["tabular"],
        temporal_csv=cfg["csv_paths"]["temporal"],
        target_col=cfg["target"],
        tab_features=cfg["tabular_features"],
        window=int(cfg["temporal_params"]["window"]),
    )
    print(f"   ✓ Samples: {len(ds)} | Nodes: {len(ds.node_mapping)} | Tab features: {len(cfg['tabular_features'])}")

    print("\n[3/10] Generating vehicle-based splits (no data leakage)...")
    plates = ds.df["num_plate"].tolist()
    uniq = np.array(sorted(set(plates)))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    test_size = float(cfg["training_params"].get("test_size", 0.2))
    val_size  = float(cfg["training_params"].get("val_size", 0.1))
    n_test = int(round(test_size * n))
    n_val  = int(round(val_size * (n - n_test)))
    test_ids  = set(uniq[:n_test])
    val_ids   = set(uniq[n_test : n_test + n_val])
    train_ids = set(uniq[n_test + n_val :])
    idx_all   = np.arange(len(ds))
    train_idx = [i for i in idx_all if plates[i] in train_ids]
    val_idx   = [i for i in idx_all if plates[i] in val_ids]
    test_idx  = [i for i in idx_all if plates[i] in test_ids]
    if not (train_idx and val_idx and test_idx):
        raise ValueError("One or more partitions are empty.")
    print(f"   ✓ Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    print("\n[4/10] Normalising tabular features (train statistics only)...")
    cols = list(cfg["tabular_features"])
    m = ds.df.loc[train_idx, cols].mean()
    s = ds.df.loc[train_idx, cols].std().replace(0, 1e-6)
    ds.df.loc[:, cols] = ((ds.df[cols] - m) / s).fillna(0.0)
    print("   ✓ Normalisation complete")

    print("\n[5/10] Creating data loaders...")
    batch = int(cfg["training_params"]["batch_size"])
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = GeoDataLoader(torch.utils.data.Subset(ds, train_idx), batch_size=batch, shuffle=True, generator=g, collate_fn=collate_three)
    val_loader   = GeoDataLoader(torch.utils.data.Subset(ds, val_idx),   batch_size=batch, shuffle=False, collate_fn=collate_three)
    test_loader  = GeoDataLoader(torch.utils.data.Subset(ds, test_idx),  batch_size=batch, shuffle=False, collate_fn=collate_three)
    print(f"   ✓ Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

    print("\n[6/10] Building GCVA model...")
    node_feat_dim = len(ds.node_mapping) + 1
    model = GCVA(node_feat_dim, 1, len(cols), cfg["gnn_model_params"], cfg["tab_model_params"], cfg["fusion_params"]).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Total parameters: {total_p:,}")

    print("\n[7/10] Configuring optimiser and loss...")
    y_train_arr = np.array([ds[i][3].item() for i in train_idx], dtype=int)
    pos_w = compute_pos_weight_from_labels(y_train_arr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training_params"]["learning_rate"]))
    print(f"   ✓ Positive weight: {pos_w:.4f} | LR: {cfg['training_params']['learning_rate']}")

    total_epochs = int(cfg["training_params"]["epochs"])
    patience     = int(cfg["training_params"]["patience"])
    lambda1 = float(cfg["fusion_params"].get("lambda1", 0.5))   # L_unc weight (paper: 0.50)
    lambda2 = float(cfg["fusion_params"].get("lambda2", 0.25))  # L_gate weight (paper: 0.25)

    best_val = float("inf")
    best_path = os.path.join(cfg["output_dir"], "best_model.pth")
    trigger = 0
    train_losses, val_losses, val_f1s = [], [], []

    print("\n[8/10] TRAINING")
    print("=" * 80)
    start = time.time()

    for epoch in range(1, total_epochs + 1):
        model.train()
        total = 0.0
        for batch_idx, (G, tab, seq, y) in enumerate(train_loader):
            G, tab, seq, y = G.to(device), tab.to(device), seq.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(G, tab, seq)
            loss, *_ = gcva_loss(out, y, criterion, lambda1, lambda2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.item())
            if (batch_idx + 1) % 50 == 0:
                print(f"   Epoch {epoch:02d} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}", flush=True)

        tr_loss = total / max(1, len(train_loader))
        train_losses.append(tr_loss)

        model.eval()
        v_total, v_probs, v_labels = 0.0, [], []
        weights_sum = torch.zeros(3, dtype=torch.float64)
        n_cnt = 0
        with torch.no_grad():
            for G, tab, seq, y in val_loader:
                G, tab, seq, y = G.to(device), tab.to(device), seq.to(device), y.to(device)
                out = model(G, tab, seq)
                loss_val, *_ = gcva_loss(out, y, criterion, lambda1, lambda2)
                v_total += float(loss_val.item())
                p = torch.sigmoid(out["logits"].view(-1))
                v_probs.extend(p.cpu().numpy().tolist())
                v_labels.extend(y.view(-1).cpu().numpy().tolist())
                weights_sum += out["fusion_weights"].sum(dim=0).double().cpu()
                n_cnt += out["fusion_weights"].size(0)

        val_loss = v_total / max(1, len(val_loader))
        val_losses.append(val_loss)
        thr = best_threshold_from_probs(np.array(v_probs), np.array(v_labels))
        f1  = f1_at_threshold(v_probs, v_labels, thr)
        val_f1s.append(f1)
        w_str = "[" + ", ".join(f"{w:.3f}" for w in (weights_sum / max(1, n_cnt)).numpy()) + "]"
        print(f"\n>>> Epoch {epoch:02d}/{total_epochs} | Train: {tr_loss:.4f} | Val: {val_loss:.4f} | F1: {f1:.4f} | Thr: {thr:.3f} | W={w_str}", flush=True)

        if val_loss < best_val:
            best_val = val_loss; trigger = 0
            torch.save(model.state_dict(), best_path)
            print(f"    ✓ New best model (val_loss: {best_val:.4f})", flush=True)
        else:
            trigger += 1
            print(f"    ⚠ No improvement: {trigger}/{patience}", flush=True)
            if trigger >= patience:
                print(f"\n>>> EARLY STOPPING at epoch {epoch}", flush=True)
                break

    train_time = time.time() - start
    print(f"\n[9/10] Training completed in {train_time:.2f}s ({train_time/60:.2f} min)")
    save_training_curves(train_losses, val_losses, val_f1s, cfg["output_dir"])

    print("\n[10/10] Test evaluation...")
    try:
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    except TypeError:
        model.load_state_dict(torch.load(best_path, map_location=device))

    model.eval()
    v_probs, v_labels = [], []
    with torch.no_grad():
        for G, tab, seq, y in val_loader:
            G, tab, seq, y = G.to(device), tab.to(device), seq.to(device), y.to(device)
            out = model(G, tab, seq)
            v_probs.extend(torch.sigmoid(out["logits"].view(-1)).cpu().numpy().tolist())
            v_labels.extend(y.view(-1).cpu().numpy().tolist())
    final_thr = best_threshold_from_probs(np.array(v_probs), np.array(v_labels))
    print(f"   ✓ Best threshold (val): {final_thr:.3f}")

    probs_t, labels_t, test_weights_list = [], [], []
    with torch.no_grad():
        for G, tab, seq, y in test_loader:
            G, tab, seq, y = G.to(device), tab.to(device), seq.to(device), y.to(device)
            out = model(G, tab, seq)
            probs_t.extend(torch.sigmoid(out["logits"].view(-1)).cpu().numpy().tolist())
            labels_t.extend(y.view(-1).cpu().numpy().tolist())
            test_weights_list.append(out["fusion_weights"].cpu().numpy())

    all_test_weights = np.concatenate(test_weights_list, axis=0)
    print("   Generating fusion weight plots...")
    plot_fusion_weights_violin(all_test_weights, cfg["output_dir"])
    plot_fusion_weights_ternary(all_test_weights, cfg["output_dir"])

    roc_auc, rep_test = save_conf_roc(labels_t, probs_t, final_thr, cfg["output_dir"], "gcva")
    print(f"   ✓ AUC: {roc_auc:.4f}")
    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT (TEST):")
    print("=" * 80)
    print(rep_test)

    with open(os.path.join(cfg["output_dir"], "summary.txt"), "w") as f:
        f.write("=== Gated Cross-View Attention (GCVA) ===\n")
        f.write(f"Training time (s): {train_time:.2f}\n")
        f.write(f"Parameters: {total_p:,}\n")
        f.write(f"Threshold (val): {final_thr:.3f}\n")
        f.write(f"AUC (test): {roc_auc:.4f}\n")
        f.write("\nClassification Report (test):\n")
        f.write(rep_test)

    print(f"\n✓ Results saved to: {cfg['output_dir']}")
    print("✓ PROCESS COMPLETED SUCCESSFULLY\n")


if __name__ == "__main__":
    main()
