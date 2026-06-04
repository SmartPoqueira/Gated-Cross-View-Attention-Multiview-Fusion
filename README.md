# GCVA: Gated Cross-View Attention for Multiview Fusion

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Journal: IEEE IoT Journal](https://img.shields.io/badge/Journal-IEEE%20IoT%20Journal-blue.svg)](https://doi.org/10.1109/JIOT.2026.3685702)

An **uncertainty-aware multiview fusion mechanism** that dynamically weights heterogeneous data views — graph-structured vehicle routes, tabular contextual features, and temporal traffic signals — based on their estimated prediction confidence. GCVA prevents noisy views from corrupting the shared feature space before cross-view attention.

---

## Overview

In real-world IoT systems, data arrives from multiple sources with different noise levels. Naively fusing them degrades performance when one view is unreliable. GCVA addresses this by:

1. **Independent view encoding** — EGAT for graph routes, TabularTransformer for contextual features, LSTM for temporal signals.
2. **Uncertainty estimation per view** — A learned head estimates aleatoric variance σ²_v; confidence is γ_v = −log(σ²_v).
3. **Confidence-aware gating** — Each view receives a scalar gate g_v = σ(W_g·[h_v; γ_v]) ∈ (0,1) based on both its embedding and its confidence.
4. **Gated cross-attention** — Attention scores between views are modulated by the product of their gates: A_ij = (Q_i K_j^T) · g_i · g_j, applied **before** softmax.
5. **Weighted fusion** — Final representation is a softmax-weighted sum of attended view embeddings, again weighted by g_v.

<p align="center">
  <img src="images/architecture.png" width="750"/>
</p>

*GCVA fusion mechanism. Each view's uncertainty σ²_v is estimated first; reliability gates g_v then modulate cross-view attention scores and the final weighted fusion.*

---

## Method

The GCVA model implements four stages (paper Section III):

| Stage | Equation | Purpose |
|---|---|---|
| 1. Uncertainty | `σ²_v = softplus(f_unc(h_v)) + ε` | Estimate aleatoric noise per view |
| 2. Gating | `g_v = σ(f_gate([h_v ; γ_v]))` | Scalar trust score ∈ (0,1) |
| 3. Cross-Attention | `A_ij = (Q_i K_j^T) · g_i · g_j` before softmax | Gated inter-view information exchange |
| 4. Fusion | `z = Σ_v ω_v · h̃_v`, `ω = softmax(g)` | Reliability-weighted combination |

**Composite loss** (paper Eq. 7):

```math
\mathcal{L} = \mathcal{L}_{\text{main}} + \lambda_1 \mathcal{L}_{\text{unc}} + \lambda_2 \mathcal{L}_{\text{gate}}
```

where λ₁ = 0.50 and λ₂ = 0.25 (selected via sensitivity analysis).

---

## Results

**Vehicle revisit prediction** — Alpujarra smart village LPR dataset, 5-fold cross-validation:

| Model | Weighted F1 ↑ | AUROC ↑ | Time (s) ↓ |
|---|---|---|---|
| Late Fusion | 0.663 | 0.712 | — |
| Concatenation | 0.679 | 0.724 | — |
| GMU | 0.712 | 0.740 | 7050 |
| Attention Fusion | 0.701 | 0.740 | 8392 |
| Cross-Att + GMU | 0.685 | 0.726 | 9211 |
| GAFN | 0.710 | 0.735 | 9539 |
| **GCVA (ours)** | **0.730** | **0.752** | **6715** |

GCVA outperforms the best baseline by **2.8%** in Weighted F1 while reducing training time by **4.7%–40.2%**.

<p align="center">
  <img src="images/results.png" width="600"/>
</p>

*Weighted F1-score distributions for fusion strategies and view combinations (5-fold CV).*

### Gate Correlation Analysis

<p align="center">
  <img src="images/gate_correlation.png" width="650"/>
</p>

*Correlation between learned gate values g_v and view prediction errors — confirming that the gating mechanism correctly suppresses less reliable views.*

### Sensitivity Analysis

<p align="center">
  <img src="images/sensitivity.png" width="600"/>
</p>

*Sensitivity of GCVA to loss hyperparameters λ₁ (uncertainty weight) and λ₂ (gate regularisation). Optimal: λ₁=0.50, λ₂=0.25.*

---

## Encoders

| View | Encoder | Why |
|---|---|---|
| Graph (vehicle route) | EGAT (Edge-featured GAT) | Edge travel times require edge-aware message passing |
| Tabular (13 features) | TabularTransformer | Per-column token embeddings capture feature interactions |
| Temporal (W=28 days) | LSTM | Captures short-term traffic demand periodicity |

All encoders project to a common embedding space d_f = 128 before fusion.

---

## Training Details (paper Section IV.C)

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| LR (encoders) | 1e-4 |
| LR (fusion) | 1e-3 |
| Batch size | 128 |
| Epochs | 100 |
| Early stopping patience | 10 |
| Temporal window W | 28 days |
| Fusion dim d_f | 128 |
| Attention heads | 4 |
| λ₁ (uncertainty) | 0.50 |
| λ₂ (gate reg.) | 0.25 |

---

## Project Structure

```
├── configs/
│   └── config.yaml          # Full hyperparameters
├── images/                  # Architecture diagrams and result figures
├── scripts/
│   └── run_experiment.sh    # Pipeline runner
└── src/
    ├── model.py             # GCVA fusion model (canonical implementation)
    ├── egat.py              # Edge-featured GAT encoder
    ├── dataset.py           # Multiview dataset and collation
    ├── data_loader.py       # Data loading utilities
    ├── training.py          # Training loop with early stopping
    ├── ablation.py          # Ablation configurations
    ├── sensitivity.py       # Sensitivity analysis
    ├── plots.py             # Fusion weight and result visualisations
    └── __init__.py
```

---

## Data

LPR camera data from the Alpujarra region (Granada, Spain). The dataset is publicly available at IEEE DataPort:  
**https://dx.doi.org/10.21227/77hy-hq94**

---

## Quick Start

```bash
pip install -r requirements.txt

python -m src.training  # requires data configured in configs/config.yaml
```

---

## Citation

This repository is published under CC BY 4.0. If you use this code, you **must** cite the paper:

```bibtex
@article{duran2026gcva,
  title={GCVA: A Multiview Fusion Mechanism for Heterogeneous Data Representations},
  author={Dur{\'a}n-L{\'o}pez, Alberto and Bola{\~n}os-Martinez, Daniel and Bermudez-Edo, Maria},
  journal={IEEE Internet of Things Journal},
  year={2026},
  publisher={IEEE}
}
```

---

## License

[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) — Copyright © 2025 SmartPoqueira.  
Free to use, adapt, and distribute for any purpose, including commercially, **provided you give appropriate credit and cite the paper above**.
