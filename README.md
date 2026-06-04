# GCVA: Gated Cross-View Attention for Multiview Fusion

An **uncertainty-aware multiview fusion mechanism** for integrating heterogeneous data views — graph-structured routes, tabular features, and temporal signals — into a unified representation for binary classification. GCVA uses gated cross-attention where each view's contribution is dynamically weighted by its estimated prediction confidence.

## Overview

In real-world IoT systems, data arrives from multiple heterogeneous sources with different noise levels and modalities. Naively concatenating these views degrades performance when one source is unreliable. GCVA addresses this by:

1. **Encoding each view independently** — EGAT for graph routes, TabularTransformer for features, BiLSTM for temporal signals.
2. **Estimating uncertainty per view** — A learned confidence score $\gamma_v$ reflects each encoder's prediction reliability.
3. **Gating cross-attention** — Attention scores between views are modulated by the product of their gate values: $\text{score}_{ij} = (Q_i K_j^T / \sqrt{d}) \cdot g_i \cdot g_j$.
4. **Weighted fusion** — Final representation is a softmax-weighted sum of attended view embeddings.


<p align="center">
  <img src="images/architecture.png" width="750"/>
</p>

*GCVA fusion mechanism. The model processes heterogeneous input views by first estimating their individual uncertainty (σ²) via the Confidence-Aware Gating Mechanism. These estimates generate reliability gates (g_v) that dynamically modulate attention scores, lowering the contribution of noisy views before the final weighted fusion.*

## Method

The GCVA model has four stages:

| Stage | Operation | Purpose |
|---|---|---|
| 1. Uncertainty | $\sigma_v = \text{softplus}(W_v h_v)$, $\gamma_v = \log(1/\sigma_v)$ | Estimate per-view confidence |
| 2. Gating | $g_v = \sigma(W_g [h_v; \gamma_v])$ | Gate views by confidence |
| 3. Cross-Attention | $\text{Attn} = \text{softmax}((QK^T/\sqrt{d}) \cdot G \cdot G^T)V$ | Inter-view information exchange |
| 4. Fusion | $z = \sum_v \omega_v \cdot h_v'$, where $\omega = \text{softmax}(g)$ | Weighted combination |

**Composite loss**: $\mathcal{L} = \mathcal{L}_\text{main} + \mu_1 \mathcal{L}_\text{aux} + \mu_2 \mathcal{L}_\text{unc} + \mu_3 \mathcal{L}_\text{gate}$

## Results

Binary classification (repeater vs. non-repeater) on IoT vehicle data:

| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| EGAT only (Graph) | 0.72 | 0.68 | 0.70 | 0.78 |
| TabTransformer only | 0.69 | 0.71 | 0.70 | 0.76 |
| BiLSTM only (Temporal) | 0.65 | 0.63 | 0.64 | 0.72 |
| Late Fusion (concat) | 0.74 | 0.72 | 0.73 | 0.81 |
| Gated Multimodal Unit | 0.76 | 0.74 | 0.75 | 0.83 |
| **GCVA (Ours)** | **0.81** | **0.78** | **0.79** | **0.87** |

<p align="center">
  <img src="images/results.png" width="700"/>
</p>

*Weighted F1-score distributions for different fusion strategies and view combinations using 5-fold cross-validation.*

### Gate Correlation Analysis

<p align="center">
  <img src="images/gate_correlation.png" width="700"/>
</p>

*Correlation between learned gate values (g_v) and view prediction errors — confirming that the gating mechanism correctly suppresses less reliable views.*

### Sensitivity Analysis

<p align="center">
  <img src="images/sensitivity.png" width="700"/>
</p>

*Sensitivity analysis of GCVA loss hyperparameters (λ₁ and λ₂).*

## Project Structure

```
GCVA/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
│   └── config.yaml          # Full hyperparameters
├── src/
│   ├── __init__.py
│   ├── model.py             # GCVA fusion model
│   ├── egat.py              # Edge-featured GAT encoder
│   ├── data_loader.py       # Multiview data loading
│   └── utils.py
├── paper/
│   ├── main.tex
│   └── figures/
└── scripts/
    └── run_experiment.sh
```

## Quick Start

```bash
pip install -r requirements.txt
python -m src.model --config configs/config.yaml
```

## Citation

```bibtex
@article{duran2026gcva,
  title={GCVA: A Multiview Fusion Mechanism for Heterogeneous Data Representations},
  author={Dur{\'a}n-L{\'O}pez, Alberto and Bola{\~n}os-Martinez, Daniel and Bermudez-Edo, Maria},
  journal={IEEE Internet of Things Journal},
  year={2026},
  publisher={IEEE}
}
```

## License

MIT License — see [LICENSE](LICENSE).
