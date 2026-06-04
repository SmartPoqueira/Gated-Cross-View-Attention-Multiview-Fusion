"""
plots.py
--------
Visualisation utilities for the GCVA pipeline:
  - Confusion matrix and ROC curve
  - Training/validation loss and F1 curves
  - Fusion-weight violin plot
  - Fusion-weight ternary plot (requires kaleido for static PNG export)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


def ensure_dir(path: str) -> None:
    """Create *path* (and all parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def save_conf_roc(y_true, y_prob, thr: float, out_dir: str, tag: str) -> tuple:
    """
    Save a confusion matrix PNG, an ROC curve PNG, and a classification report.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities.
        thr (float): Decision threshold.
        out_dir (str): Output directory.
        tag (str): Filename prefix.

    Returns:
        tuple: (roc_auc, classification_report_string)
    """
    y_true = np.asarray(y_true, int)
    y_prob = np.asarray(y_prob, float)
    y_pred = (y_prob > thr).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix ({tag})")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-Rep", "Rep"]); ax.set_yticklabels(["Non-Rep", "Rep"])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    ensure_dir(out_dir)
    fig.savefig(os.path.join(out_dir, f"cm_{tag}.png"))
    plt.close(fig)

    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "--")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"roc_{tag}.png"))
        plt.close()
    except Exception:
        roc_auc = float("nan")

    rep = classification_report(y_true, y_pred, target_names=["Non-Rep", "Rep"], digits=4)
    with open(os.path.join(out_dir, f"classification_report_{tag}.txt"), "w") as f:
        f.write(rep)

    return roc_auc, rep


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def save_training_curves(
    train_losses: list, val_losses: list, val_f1s: list, out_dir: str
) -> None:
    """Save loss and F1-score curves over training epochs."""
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, train_losses, "b-", label="Train Loss")
    ax1.plot(epochs, val_losses, "r-", label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, val_f1s, "g-", label="Val F1")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("F1 Score")
    ax2.set_title("Validation F1 Score"); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    ensure_dir(out_dir)
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Fusion weight distribution plots
# ---------------------------------------------------------------------------

def plot_fusion_weights_violin(
    all_weights: np.ndarray,
    output_dir: str,
    view_names: list = None,
) -> None:
    """
    Violin plot showing the distribution of fusion weights ω_v for each view.

    Args:
        all_weights (np.ndarray): Shape (N, 3) — fusion weights per sample.
        output_dir (str): Output directory.
        view_names (list): Names for the three views.
    """
    if view_names is None:
        view_names = ["Graph", "Tabular", "Temporal"]
    import pandas as pd
    df_w = pd.DataFrame(all_weights, columns=view_names)
    df_melt = df_w.melt(var_name="View", value_name="Fusion Weight")

    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(8, 6))
    sns.violinplot(
        x="View", y="Fusion Weight", data=df_melt,
        palette=["#e74c3c", "#3498db", "#2ecc71"],
        cut=0, inner="box", alpha=0.8,
    )
    plt.title("Distribution of Fusion Weights ω_v (Test Set)", fontsize=14, fontweight="bold")
    plt.ylabel("Fusion Weight (ω)", fontsize=12)
    plt.xlabel("")
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    ensure_dir(output_dir)
    plt.savefig(os.path.join(output_dir, "fusion_weights_violinplot.png"), dpi=300)
    plt.close()
    print("   ✓ Fusion weight violin plot saved.")


def plot_fusion_weights_ternary(
    all_weights: np.ndarray,
    output_dir: str,
    view_names: list = None,
) -> None:
    """
    Ternary scatter plot of fusion weights (requires plotly + kaleido).

    Falls back to an HTML file if kaleido is unavailable.

    Args:
        all_weights (np.ndarray): Shape (N, 3).
        output_dir (str): Output directory.
        view_names (list): Names for the three views.
    """
    if view_names is None:
        view_names = ["Graph", "Tabular", "Temporal"]
    if all_weights.shape[1] != 3:
        print("   ⚠ Ternary plot requires exactly 3 views. Skipping.")
        return

    try:
        import plotly.express as px
        import plotly.io as pio
        import pandas as pd
    except ImportError:
        print("   ⚠ plotly not installed. Skipping ternary plot.")
        return

    df_w = pd.DataFrame(all_weights, columns=view_names)
    fig = px.scatter_ternary(
        df_w, a=view_names[0], b=view_names[1], c=view_names[2],
        color_discrete_sequence=["#2ecc71"],
        title="Ternary Plot of Fusion Weights (Test Set)",
        opacity=0.6,
    )
    fig.update_layout(
        ternary={
            "sum": 1,
            "aaxis": {"title": view_names[0], "min": 0.01, "linewidth": 2, "ticks": "outside"},
            "baxis": {"title": view_names[1], "min": 0.01, "linewidth": 2, "ticks": "outside"},
            "caxis": {"title": view_names[2], "min": 0.01, "linewidth": 2, "ticks": "outside"},
        },
        font=dict(size=14),
        margin=dict(t=80, b=40, l=40, r=40),
    )
    ensure_dir(output_dir)
    try:
        pio.write_image(fig, os.path.join(output_dir, "fusion_weights_ternaryplot.png"), scale=3)
        print("   ✓ Ternary plot saved (PNG).")
    except Exception as e:
        print(f"   ⚠ Static PNG export failed ({e}). Saving HTML instead.")
        fig.write_html(os.path.join(output_dir, "fusion_weights_ternaryplot.html"))
        print("   ✓ Ternary plot saved (HTML).")
