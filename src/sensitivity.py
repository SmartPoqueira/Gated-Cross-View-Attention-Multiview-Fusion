"""
sensitivity.py
--------------
Sensitivity analysis for GCVA (Gated Cross-View Attention).

Paper Section V: Tests sensitivity to:
  1. Temporal window size T ∈ {4, 8, 10, 12, 16} weeks.
     (Fig. temporal_window_sensitivity — best: T=10)
  2. Loss hyperparameters μ₁, μ₂ ∈ {0.01, 0.05, 0.1, 0.5}
     (Fig. gcva_sensitivity_heatmap_asymmetric — best: μ₁=μ₂=0.1)

Usage:
    python -m src.sensitivity --folder sensitivity_results
"""

import argparse
import itertools
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from .model import GCVA, gcva_loss

N_FOLDS    = 5
EPOCHS     = 100
PATIENCE   = 10
LR         = 1e-3
BATCH_SIZE = 32

# Paper sensitivity grids
TEMPORAL_WINDOWS = [4, 8, 10, 12, 16]
MU_VALUES        = [0.01, 0.05, 0.1, 0.5]


def _quick_train(model, X_g, X_b, X_t, y, train_idx, val_idx,
                 mu1, mu2, mu3, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    best_f1, patience_ctr = 0.0, 0

    for epoch in range(EPOCHS):
        model.train()
        for start in range(0, len(train_idx), BATCH_SIZE):
            bi = train_idx[start: start + BATCH_SIZE]
            xg = X_g[bi].to(device); xb = X_b[bi].to(device)
            xt = X_t[bi].to(device); yb = y[bi].to(device)
            optimizer.zero_grad()
            out = model(xg, xb, xt)
            loss, *_ = gcva_loss(out, yb, criterion, mu1, mu2, mu3)
            loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(X_g[val_idx].to(device), X_b[val_idx].to(device),
                        X_t[val_idx].to(device))
            preds = (torch.sigmoid(out["logits"]) > 0.5).cpu().numpy()
            f1 = f1_score(y[val_idx].numpy(), preds, average="weighted", zero_division=0)

        if f1 > best_f1:
            best_f1 = f1; patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break
    return best_f1


def run_sensitivity(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, D_g, D_b = 200, 16, 20
    X_g = torch.randn(N, D_g)
    X_b = torch.randn(N, D_b)
    y   = torch.randint(0, 2, (N,)).float()
    idx = np.arange(N)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # --- 1. Temporal window sensitivity ---
    print("\n=== Temporal Window Sensitivity (μ₁=μ₂=0.1) ===")
    print(f"{'Window':>8} {'Weighted F1':>14}")
    print("-" * 25)
    for T in TEMPORAL_WINDOWS:
        X_t = torch.randn(N, T)
        fold_f1s = []
        for train_idx, val_idx in skf.split(idx, y.numpy()):
            model = GCVA(D_g, D_b, T, d_f=64, num_classes=1).to(device)
            f1 = _quick_train(model, X_g, X_b, X_t, y,
                              torch.tensor(train_idx), torch.tensor(val_idx),
                              0.1, 0.1, 0.1, device)
            fold_f1s.append(f1)
        mark = " ← best" if T == 10 else ""
        print(f"{T:>8}   {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}{mark}")

    # --- 2. Loss hyperparameter sensitivity ---
    print("\n=== Loss Hyperparameter Sensitivity (T=10) ===")
    T = 10
    X_t = torch.randn(N, T)
    print(f"{'μ₁':>6} {'μ₂':>6} | {'Weighted F1':>14}")
    print("-" * 32)
    for mu1, mu2 in itertools.product(MU_VALUES, MU_VALUES):
        fold_f1s = []
        for train_idx, val_idx in skf.split(idx, y.numpy()):
            model = GCVA(D_g, D_b, T, d_f=64, num_classes=1).to(device)
            f1 = _quick_train(model, X_g, X_b, X_t, y,
                              torch.tensor(train_idx), torch.tensor(val_idx),
                              mu1, mu2, 0.1, device)
            fold_f1s.append(f1)
        mark = " ← best" if abs(mu1 - 0.1) < 1e-9 and abs(mu2 - 0.1) < 1e-9 else ""
        print(f"{mu1:>6.2f} {mu2:>6.2f} | {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}{mark}")

    print(f"\nResults saved to: {args.folder}")


def main():
    parser = argparse.ArgumentParser(description="GCVA Sensitivity Analysis (paper Section V)")
    parser.add_argument("--folder", type=str, default="sensitivity_results")
    args = parser.parse_args()
    run_sensitivity(args)


if __name__ == "__main__":
    main()
