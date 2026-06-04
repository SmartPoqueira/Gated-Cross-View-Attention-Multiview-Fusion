"""
model.py
--------
Neural network architectures for the GCVA (Gated Cross-View Attention)
multiview fusion model.

The model integrates three heterogeneous views:
  - Graph view   : vehicle routes as directed graphs (EGAT)
  - Tabular view : IoT tabular features (TabularTransformer)
  - Temporal view: daily traffic counts (BiLSTM)

Architecture overview (see paper/images/gcva_updated.png):
  1. Encode each view independently → h_g, h_b, h_t ∈ R^d_f
  2. Estimate per-view uncertainty σ_v² and confidence γ_v = log(1/σ_v)
  3. Compute reliability gate g_v = σ(W_g · [h_v; γ_v]) ∈ (0,1)
  4. Gated cross-attention: score_ij = (Q_i K_j^T / √d) · g_i · g_j
  5. Weighted fusion: z = Σ_v ω_v · h_v',  ω = softmax(g)
  6. Binary classification head on z

Composite loss:
  L = L_main + μ₁·L_aux + μ₂·L_unc + μ₃·L_gate

References
----------
Durán-López et al., IEEE Internet of Things Journal, 2026.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .egat import EGAT


# ---------------------------------------------------------------------------
# View encoders
# ---------------------------------------------------------------------------

class GraphEncoderFromEGAT(nn.Module):
    """
    Graph encoder wrapping the EGAT module.

    The EGAT final linear layer is replaced by an identity to expose the
    raw node embeddings for downstream fusion.

    Args:
        num_node_features (int): Dimensionality of node feature vectors.
        num_edge_features (int): Dimensionality of edge feature vectors.
        model_params (dict): EGAT hyperparameters.
    """

    def __init__(self, num_node_features: int, num_edge_features: int, model_params: dict) -> None:
        super().__init__()
        self.egat = EGAT(num_node_features, num_edge_features, model_params)
        if hasattr(self.egat, "final_linear"):
            self.egat.final_linear = nn.Identity()

    def forward(self, x, edge_index, edge_attr, batch):
        return self.egat(x, edge_index, edge_attr, batch)


class TabularTransformer(nn.Module):
    """
    Tabular feature encoder based on a Transformer with per-column token embeddings.

    Each feature value is projected to a d_model-dimensional token, augmented with
    a learnable column embedding.  A [CLS] token (optional) summarises the sequence.

    Args:
        in_features (int): Number of tabular input features.
        d_model (int): Token embedding dimension.
        n_heads (int): Transformer attention heads.
        n_layers (int): Transformer encoder layers.
        dim_ff (int): Feed-forward hidden dimension.
        dropout (float): Dropout rate.
        use_cls (bool): Use CLS token pooling (vs. mean pooling).
    """

    def __init__(
        self,
        in_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_ff: int = 128,
        dropout: float = 0.2,
        use_cls: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.d_model = d_model
        self.use_cls = use_cls
        self.value_proj = nn.Linear(1, d_model)
        self.col_embed = nn.Parameter(torch.randn(1, in_features, d_model) * 0.02)
        if use_cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pre_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the d_model-dimensional representation."""
        B, _ = x.shape
        toks = self.value_proj(x.unsqueeze(-1)) + self.col_embed
        toks = self.pre_drop(toks)
        if self.use_cls:
            toks = torch.cat([self.cls_token.expand(B, 1, -1), toks], dim=1)
        z = self.norm(self.encoder(toks))
        return z[:, 0, :] if self.use_cls else z.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x)).view(-1)


class LSTMTemporal(nn.Module):
    """
    Temporal sequence encoder based on a (Bi-)LSTM.

    The final hidden state is projected through a linear layer and returned
    as a fixed-size embedding.

    Args:
        hidden_dim (int): LSTM hidden units per direction.
        bidirectional (bool): Use bidirectional LSTM.
        dropout (float): Dropout on the projection layer.
    """

    def __init__(self, hidden_dim: int, bidirectional: bool = False, dropout: float = 0.2) -> None:
        super().__init__()
        self.bidirectional = bool(bidirectional)
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden_dim,
            batch_first=True, bidirectional=self.bidirectional,
        )
        out_dim = hidden_dim * (2 if self.bidirectional else 1)
        self.fc = nn.Linear(out_dim, out_dim)
        self.head = nn.Linear(out_dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, seq: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        x = seq.unsqueeze(-1)
        _, (hn, _) = self.lstm(x)
        h = hn.transpose(0, 1).reshape(seq.size(0), -1)
        z = self.drop(F.relu(self.fc(h)))
        if return_embedding:
            return z
        return self.head(z).view(-1)


# ---------------------------------------------------------------------------
# GCVA — Gated Cross-View Attention
# ---------------------------------------------------------------------------

class GCVA(nn.Module):
    """
    Gated Cross-View Attention fusion model.

    Implements all four stages described in the paper:
      1. Uncertainty estimation via softplus
      2. Confidence-Aware Gating
      3. Gated Cross-Attention (masking diagonal to force cross-view exchange)
      4. Softmax-weighted fusion

    Args:
        node_feat_dim (int): Node feature dimension for EGAT.
        edge_feat_dim (int): Edge feature dimension for EGAT.
        tab_in (int): Number of tabular input features.
        gnn_params (dict): EGAT hyperparameters.
        tab_params (dict): TabularTransformer hyperparameters.
        fusion_params (dict): Fusion hyperparameters (fusion_dim, tau, mu_*, temporal_*).
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        tab_in: int,
        gnn_params: dict,
        tab_params: dict,
        fusion_params: dict,
    ) -> None:
        super().__init__()

        # --- View encoders ---
        self.g_enc = GraphEncoderFromEGAT(node_feat_dim, edge_feat_dim, gnn_params)
        d_model   = int(tab_params.get("d_model", tab_params.get("hidden_dim", 64)))
        n_heads   = int(tab_params.get("n_heads", 4))
        n_layers  = int(tab_params.get("n_layers", 2))
        dim_ff    = int(tab_params.get("dim_ff", 128))
        drop_tt   = float(tab_params.get("dropout", 0.2))
        use_cls   = bool(tab_params.get("use_cls", True))
        self.b_enc = TabularTransformer(tab_in, d_model, n_heads, n_layers, dim_ff, drop_tt, use_cls)
        hidden_t  = int(fusion_params.get("temporal_hidden", 64))
        bidir_t   = bool(fusion_params.get("temporal_bidirectional", False))
        drop_t    = float(fusion_params.get("temporal_dropout", 0.2))
        self.t_enc = LSTMTemporal(hidden_dim=hidden_t, bidirectional=bidir_t, dropout=drop_t)

        # --- Projection to common fusion dimension d_f ---
        hidden_g = int(gnn_params.get("hidden_dim", 64))
        out_dim_t = hidden_t * (2 if bidir_t else 1)
        self.d_f = int(fusion_params.get("fusion_dim", 64))

        self.proj_g = nn.Linear(hidden_g, self.d_f)
        self.proj_b = nn.Linear(d_model, self.d_f)
        self.proj_t = nn.Linear(out_dim_t, self.d_f)

        # --- Stage 1: Uncertainty estimation σ_v ---
        self.uncertainty_g = nn.Linear(self.d_f, 1)
        self.uncertainty_b = nn.Linear(self.d_f, 1)
        self.uncertainty_t = nn.Linear(self.d_f, 1)

        # --- Stage 2: Gate networks g_v = σ(W_g · [h_v; γ_v]) ---
        def _gate_net():
            return nn.Sequential(
                nn.Linear(self.d_f + 1, self.d_f // 2), nn.ReLU(),
                nn.Linear(self.d_f // 2, 1), nn.Sigmoid(),
            )
        self.gate_net_g = _gate_net()
        self.gate_net_b = _gate_net()
        self.gate_net_t = _gate_net()

        # --- Stage 3: Cross-attention projections ---
        self.W_Q = nn.Linear(self.d_f, self.d_f)
        self.W_K = nn.Linear(self.d_f, self.d_f)
        self.W_V = nn.Linear(self.d_f, self.d_f)

        # --- Stage 4: Fusion head and auxiliary classifiers ---
        self.fusion_head = nn.Sequential(
            nn.Linear(self.d_f, self.d_f // 2), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(self.d_f // 2, 1),
        )
        self.aux_g = nn.Linear(self.d_f, 1)
        self.aux_b = nn.Linear(self.d_f, 1)
        self.aux_t = nn.Linear(self.d_f, 1)

        self.tau = float(fusion_params.get("tau", 1.0))

    def compute_gates(self, h_g, h_b, h_t):
        """
        Compute per-view uncertainty σ_v and gate g_v.

        Stage 1: σ_v = softplus(W_σ h_v) + ε
        Stage 2: γ_v = log(1 / σ_v)   (log-confidence)
                 g_v = σ(W_g · [h_v; γ_v])

        Returns:
            gates:        (B, 3)  — one gate per view
            uncertainties:(B, 3)  — one σ per view
        """
        unc_g = F.softplus(self.uncertainty_g(h_g)) + 1e-6
        unc_b = F.softplus(self.uncertainty_b(h_b)) + 1e-6
        unc_t = F.softplus(self.uncertainty_t(h_t)) + 1e-6
        conf_g = torch.log(1.0 / torch.clamp(unc_g, min=1e-4))
        conf_b = torch.log(1.0 / torch.clamp(unc_b, min=1e-4))
        conf_t = torch.log(1.0 / torch.clamp(unc_t, min=1e-4))
        gate_g = self.gate_net_g(torch.cat([h_g, conf_g], dim=1))
        gate_b = self.gate_net_b(torch.cat([h_b, conf_b], dim=1))
        gate_t = self.gate_net_t(torch.cat([h_t, conf_t], dim=1))
        return (
            torch.cat([gate_g, gate_b, gate_t], dim=1),
            torch.cat([unc_g,  unc_b,  unc_t],  dim=1),
        )

    def gated_cross_attention(self, H, gates):
        """
        Gated cross-attention across the three views.

        score_ij = (Q_i K_j^T / √d_f) · g_i · g_j

        The diagonal is masked to force cross-view (not self-) attention.

        Args:
            H: (B, 3, d_f) — stacked view embeddings.
            gates: (B, 3)

        Returns:
            H_updated: (B, 3, d_f)
            attn: (B, 3, 3)
        """
        B = H.size(0)
        Q, K, V = self.W_Q(H), self.W_K(H), self.W_V(H)
        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(self.d_f)
        gate_matrix = gates.unsqueeze(2) * gates.unsqueeze(1)   # (B, 3, 3)
        scores = scores * gate_matrix
        mask = torch.eye(3, device=H.device).unsqueeze(0).expand(B, -1, -1).bool()
        scores = scores.masked_fill(mask, -1e9)
        attn = F.softmax(scores / self.tau, dim=-1)
        H_attended = torch.matmul(attn, V)
        return H + H_attended, attn

    def forward(self, G, tab, seq):
        """
        Args:
            G   : PyG Batch — graph view.
            tab : (B, n_tab) — tabular view.
            seq : (B, window) — temporal view.

        Returns:
            dict with keys: logits, aux_logits, uncertainties, gates,
                            fusion_weights, attn_weights, z_fused.
        """
        g_emb = self.g_enc(G.x, G.edge_index, G.edge_attr, G.batch)
        b_emb = self.b_enc.encode(tab)
        t_emb = self.t_enc(seq, return_embedding=True)

        h_g, h_b, h_t = self.proj_g(g_emb), self.proj_b(b_emb), self.proj_t(t_emb)
        gates, uncertainties = self.compute_gates(h_g, h_b, h_t)

        H = torch.stack([h_g, h_b, h_t], dim=1)               # (B, 3, d_f)
        H_updated, attn_weights = self.gated_cross_attention(H, gates)

        fusion_weights = F.softmax(gates, dim=1)               # (B, 3)
        z_fused = (fusion_weights.unsqueeze(-1) * H_updated).sum(dim=1)  # (B, d_f)
        logit = self.fusion_head(z_fused).squeeze(-1)

        aux_g = self.aux_g(H_updated[:, 0]).squeeze(-1)
        aux_b = self.aux_b(H_updated[:, 1]).squeeze(-1)
        aux_t = self.aux_t(H_updated[:, 2]).squeeze(-1)

        return {
            "logits": logit,
            "aux_logits": torch.stack([aux_g, aux_b, aux_t], dim=1),
            "uncertainties": uncertainties,
            "gates": gates,
            "fusion_weights": fusion_weights,
            "attn_weights": attn_weights,
            "z_fused": z_fused,
        }


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------

def gcva_loss(
    outputs: dict,
    y: torch.Tensor,
    criterion,
    mu_aux: float = 0.3,
    mu_unc: float = 0.1,
    mu_gate: float = 0.05,
) -> tuple:
    """
    Composite GCVA loss:
        L = L_main + μ₁·L_aux + μ₂·L_unc + μ₃·L_gate

    - L_main : binary cross-entropy on the fused prediction.
    - L_aux  : mean auxiliary BCE loss across the three view-specific heads.
    - L_unc  : MAE between predicted uncertainty and observed per-view squared error
               (encourages calibrated uncertainty estimates).
    - L_gate : gate regularisation  g_v·(1-g_v) encourages binary (decisive) gates.

    Args:
        outputs (dict): GCVA.forward() output dictionary.
        y (torch.Tensor): Ground-truth binary labels (B,).
        criterion: BCEWithLogitsLoss instance (with optional pos_weight).
        mu_aux, mu_unc, mu_gate (float): Loss weighting coefficients.

    Returns:
        tuple: (L_total, L_main.item(), L_aux.item(), L_unc.item(), L_gate.item())
    """
    L_main = criterion(outputs["logits"], y)
    L_aux = (
        criterion(outputs["aux_logits"][:, 0], y)
        + criterion(outputs["aux_logits"][:, 1], y)
        + criterion(outputs["aux_logits"][:, 2], y)
    ) / 3.0
    aux_probs = torch.sigmoid(outputs["aux_logits"])
    errors = (aux_probs - y.unsqueeze(1)).pow(2)
    L_unc = (outputs["uncertainties"] - errors).abs().mean()
    L_gate = (outputs["gates"] * (1 - outputs["gates"])).mean()
    L_total = L_main + mu_aux * L_aux + mu_unc * L_unc + mu_gate * L_gate
    return L_total, L_main.item(), L_aux.item(), L_unc.item(), L_gate.item()