#!/usr/bin/env python3
import os, ast, time, yaml, math
import numpy as np, pandas as pd
from datetime import timedelta
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GeoDataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

# --- LIBRERÍAS AÑADIDAS PARA PLOTS ---
import seaborn as sns
import plotly.express as px
import plotly.io as pio
# Configuración para imágenes estáticas
try:
    import kaleido
except ImportError:
    pass # Se manejará si falla al guardar

from models.EGAT import EGAT

def read_yaml(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        return yaml.safe_load(f)

def require_keys(d, keys, root="cfg"):
    for k in keys:
        if k not in d:
            raise KeyError(f"Falta clave '{k}' en {root}")

def set_deterministic_seed(seed:int):
    import random, os as _os
    _os.environ["CUBLAS_WORKSPACE_CONFIG"]= ":4096:8"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def compute_pos_weight_from_labels(y_np: np.ndarray) -> float:
    y = np.asarray(y_np, dtype=int)
    pos = int((y==1).sum()); neg = int((y==0).sum())
    return 1.0 if pos==0 else float(neg/max(1,pos))

def best_threshold_from_probs(probs, labels) -> float:
    pr, rc, th = precision_recall_curve(labels, probs)
    f1 = 2*(pr*rc)/(pr+rc+1e-6)
    if f1.size==0: return 0.5
    idx = int(np.argmax(f1))
    return 0.5 if idx>=len(th) else float(th[idx])

def f1_at_threshold(probs, labels, thr: float) -> float:
    p = (np.asarray(probs) > thr).astype(int)
    y = np.asarray(labels).astype(int)
    tp = int(((p==1)&(y==1)).sum()); fp = int(((p==1)&(y==0)).sum()); fn = int(((p==0)&(y==1)).sum())
    prec = tp/max(1,tp+fp); rec = tp/max(1,tp+fn)
    return float(2*prec*rec/(prec+rec+1e-6))

class ThreeViewsDataset(Dataset):
    def __init__(self, graphs_csv, tabular_csv, temporal_csv, target_col, tab_features, window=7):
        gdf = pd.read_csv(graphs_csv, low_memory=False)
        tdf = pd.read_csv(tabular_csv, low_memory=False)
        tmp = pd.read_csv(temporal_csv, low_memory=False)
        if 'entry_date' not in gdf.columns:
            raise KeyError("graphs_csv requiere 'entry_date'")
        gdf['entry_date'] = pd.to_datetime(gdf['entry_date'], errors='coerce')
        if 'num_plate' not in gdf.columns or 'num_plate' not in tdf.columns:
            raise KeyError("Ambos CSVs deben contener 'num_plate'")
        gdf['num_plate'] = gdf['num_plate'].astype(str).str.strip().str.lower()
        tdf['num_plate'] = tdf['num_plate'].astype(str).str.strip().str.lower()
        if target_col not in gdf.columns:
            if 'num_visits' in gdf.columns:
                gdf[target_col] = (gdf['num_visits']>=2).astype(int)
            else:
                raise KeyError(f"No existe '{target_col}' ni 'num_visits'")
        gdf[target_col] = gdf[target_col].astype(int)
        merged = pd.merge(gdf, tdf, on='num_plate', how='inner', suffixes=('', '_tab'))
        if merged.empty:
            raise ValueError("Merge vacío")
        self.tab_features = list(tab_features)
        for c in self.tab_features:
            if c not in merged.columns:
                raise KeyError(f"Falta feature tabular '{c}'")
            merged[c] = pd.to_numeric(merged[c], errors='coerce')
        if 'day' not in tmp.columns or 'count' not in tmp.columns:
            raise KeyError("temporal_csv requiere 'day' y 'count'")
        tmp['day'] = pd.to_datetime(tmp['day'])
        tmp['count'] = pd.to_numeric(tmp['count'], errors='coerce')
        tmp['count'] = (tmp['count']-tmp['count'].mean())/(tmp['count'].std()+1e-6)
        self.temporal = tmp.set_index('day')['count'].sort_index()
        self.df = merged.reset_index(drop=True)
        self.target_col = target_col
        self.window = int(window)
        node_set = set()
        for r in self.df['route']:
            for n in ast.literal_eval(r):
                node_set.add(n)
        self.node_mapping = {n:i for i,n in enumerate(sorted(node_set))}

    def __len__(self): return len(self.df)

    def _seq_from_temporal(self, day):
        days = [(pd.to_datetime(day).normalize()-timedelta(days=i)) for i in reversed(range(self.window))]
        return torch.tensor([float(self.temporal.get(d,0.0)) for d in days], dtype=torch.float32)

    def _row_to_graph(self, row):
        route = ast.literal_eval(row['route'])
        times = ast.literal_eval(row['times'])
        directions = ast.literal_eval(row['directions'])
        n = len(route)
        if n > 0 and len(times) != max(0, n-1):
            times = times[:max(0, n-1)]
        if len(directions) != n:
            directions = directions[:n] + [0]*(n - len(directions))
        ei = [[i,i+1] for i in range(n-1)]
        if len(ei) > 0:
            t = np.asarray(times, dtype=np.float32)
            if t.size:
                t = (t - t.mean()) / (t.std() + 1e-6)
            edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
            edge_attr  = torch.tensor(t[:, None], dtype=torch.float32)
        else:
            edge_index = torch.empty((2,0), dtype=torch.long)
            edge_attr  = torch.empty((0,1), dtype=torch.float32)
        X = torch.zeros((n, len(self.node_mapping)), dtype=torch.float32)
        for i,node in enumerate(route):
            X[i, self.node_mapping[node]] = 1.0
        dir_feat = torch.tensor(directions[:n], dtype=torch.float32).unsqueeze(-1)
        X = torch.cat([X, dir_feat], dim=1)
        y = torch.tensor([row[self.target_col]], dtype=torch.float32)
        return Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=y)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        G = self._row_to_graph(row)
        tab_vals = torch.tensor(row[self.tab_features].values.astype(np.float32), dtype=torch.float32)
        seq = self._seq_from_temporal(row['entry_date'])
        y = torch.tensor(row[self.target_col], dtype=torch.float32)
        return G, tab_vals, seq, y

def collate_three(batch):
    graphs, tabs, seqs, labels = zip(*batch)
    return Batch.from_data_list(graphs), torch.stack(tabs,0), torch.stack(seqs,0), torch.stack(labels,0).view(-1)

class GraphEncoderFromEGAT(nn.Module):
    def __init__(self, num_node_features, num_edge_features, model_params):
        super().__init__()
        self.egat = EGAT(num_node_features, num_edge_features, model_params)
        if hasattr(self.egat, "final_linear"):
            self.egat.final_linear = nn.Identity()
    def forward(self, x, edge_index, edge_attr, batch):
        return self.egat(x, edge_index, edge_attr, batch)

class TabularTransformer(nn.Module):
    def __init__(self,in_features,d_model=64,n_heads=4,n_layers=2,dim_ff=128,dropout=0.2,use_cls=True):
        super().__init__()
        self.in_features=in_features
        self.d_model=d_model
        self.use_cls=use_cls
        self.value_proj=nn.Linear(1,d_model)
        self.col_embed=nn.Parameter(torch.randn(1,in_features,d_model)*0.02)
        if use_cls:
            self.cls_token=nn.Parameter(torch.randn(1,1,d_model)*0.02)
        enc_layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=n_heads,dim_feedforward=dim_ff,dropout=dropout,batch_first=True,activation="gelu")
        self.encoder=nn.TransformerEncoder(enc_layer,num_layers=n_layers)
        self.pre_drop=nn.Dropout(dropout)
        self.norm=nn.LayerNorm(d_model)
        self.head=nn.Linear(d_model,1)
    def encode(self,x:torch.Tensor)->torch.Tensor:
        B,D=x.shape
        toks=self.value_proj(x.unsqueeze(-1))
        toks=toks+self.col_embed
        toks=self.pre_drop(toks)
        if self.use_cls:
            cls=self.cls_token.expand(B,1,-1)
            toks=torch.cat([cls,toks],dim=1)
        z=self.encoder(toks)
        z=self.norm(z)
        h=z[:,0,:] if self.use_cls else z.mean(dim=1)
        return h
    def forward(self,x:torch.Tensor)->torch.Tensor:
        h=self.encode(x)
        return self.head(h).view(-1)

class LSTMTemporal(nn.Module):
    def __init__(self,hidden_dim,bidirectional=False,dropout=0.2):
        super().__init__()
        self.bidirectional=bool(bidirectional)
        self.lstm=nn.LSTM(input_size=1,hidden_size=hidden_dim,batch_first=True,bidirectional=self.bidirectional)
        out_dim=hidden_dim*(2 if self.bidirectional else 1)
        self.fc=nn.Linear(out_dim,out_dim)
        self.head=nn.Linear(out_dim,1)
        self.drop=nn.Dropout(dropout)
    def forward(self,seq,return_embedding:bool=False):
        x=seq.unsqueeze(-1)
        _,(hn,_) = self.lstm(x)
        h=hn.transpose(0,1).reshape(seq.size(0),-1)
        z=F.relu(self.fc(h))
        z=self.drop(z)
        if return_embedding:
            return z
        return self.head(z).view(-1)

class GCVA(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, tab_in, 
                 gnn_params, tab_params, fusion_params):
        super().__init__()
        # Encoders
        self.g_enc = GraphEncoderFromEGAT(node_feat_dim, edge_feat_dim, gnn_params)
        d_model = int(tab_params.get('d_model', tab_params.get('hidden_dim', 64)))
        n_heads = int(tab_params.get('n_heads', 4))
        n_layers = int(tab_params.get('n_layers', 2))
        dim_ff = int(tab_params.get('dim_ff', 128))
        drop_tt = float(tab_params.get('dropout', 0.2))
        use_cls_tt = bool(tab_params.get('use_cls', True))
        self.b_enc = TabularTransformer(tab_in, d_model, n_heads, n_layers, dim_ff, drop_tt, use_cls_tt)
        hidden_t = int(fusion_params.get('temporal_hidden', 64))
        bidir_t = bool(fusion_params.get('temporal_bidirectional', False))
        drop_t = float(fusion_params.get('temporal_dropout', 0.2))
        self.t_enc = LSTMTemporal(hidden_dim=hidden_t, bidirectional=bidir_t, dropout=drop_t)
        
        # Dimensiones
        hidden_g = int(gnn_params.get('hidden_dim', 64))
        hidden_b = d_model
        out_dim_t = hidden_t * (2 if bidir_t else 1)
        self.d_f = int(fusion_params.get('fusion_dim', 64))
        
        # Proyecciones
        self.proj_g = nn.Linear(hidden_g, self.d_f)
        self.proj_b = nn.Linear(hidden_b, self.d_f)
        self.proj_t = nn.Linear(out_dim_t, self.d_f)
        
        # Stage 1: Uncertainty estimation
        self.uncertainty_g = nn.Linear(self.d_f, 1)
        self.uncertainty_b = nn.Linear(self.d_f, 1)
        self.uncertainty_t = nn.Linear(self.d_f, 1)
        
        # Stage 2: Gate networks
        self.gate_net_g = nn.Sequential(nn.Linear(self.d_f + 1, self.d_f // 2), nn.ReLU(), nn.Linear(self.d_f // 2, 1), nn.Sigmoid())
        self.gate_net_b = nn.Sequential(nn.Linear(self.d_f + 1, self.d_f // 2), nn.ReLU(), nn.Linear(self.d_f // 2, 1), nn.Sigmoid())
        self.gate_net_t = nn.Sequential(nn.Linear(self.d_f + 1, self.d_f // 2), nn.ReLU(), nn.Linear(self.d_f // 2, 1), nn.Sigmoid())
        
        # Stage 3: Cross-attention
        self.W_Q = nn.Linear(self.d_f, self.d_f)
        self.W_K = nn.Linear(self.d_f, self.d_f)
        self.W_V = nn.Linear(self.d_f, self.d_f)
        
        # Stage 4: Fusion head
        self.fusion_head = nn.Sequential(nn.Linear(self.d_f, self.d_f // 2), nn.ReLU(), nn.Dropout(0.2), nn.Linear(self.d_f // 2, 1))
        
        # Auxiliary heads
        self.aux_g = nn.Linear(self.d_f, 1)
        self.aux_b = nn.Linear(self.d_f, 1)
        self.aux_t = nn.Linear(self.d_f, 1)
        
        # Hyperparameters
        self.tau = float(fusion_params.get('tau', 1.0))
        
    def compute_gates(self, h_g, h_b, h_t):
        unc_g = F.softplus(self.uncertainty_g(h_g)) + 1e-6
        unc_b = F.softplus(self.uncertainty_b(h_b)) + 1e-6
        unc_t = F.softplus(self.uncertainty_t(h_t)) + 1e-6
        conf_g = torch.log(1.0 / torch.clamp(unc_g, min=1e-4))
        conf_b = torch.log(1.0 / torch.clamp(unc_b, min=1e-4))
        conf_t = torch.log(1.0 / torch.clamp(unc_t, min=1e-4))
        gate_g = self.gate_net_g(torch.cat([h_g, conf_g], dim=1))
        gate_b = self.gate_net_b(torch.cat([h_b, conf_b], dim=1))
        gate_t = self.gate_net_t(torch.cat([h_t, conf_t], dim=1))
        uncertainties = torch.cat([unc_g, unc_b, unc_t], dim=1)
        gates = torch.cat([gate_g, gate_b, gate_t], dim=1)
        return gates, uncertainties
        
    def gated_cross_attention(self, H, gates):
        B = H.size(0)
        Q, K, V = self.W_Q(H), self.W_K(H), self.W_V(H)
        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(self.d_f)
        gate_matrix = gates.unsqueeze(2) * gates.unsqueeze(1)
        scores = scores * gate_matrix
        mask = torch.eye(3, device=H.device).unsqueeze(0).expand(B, -1, -1).bool()
        scores = scores.masked_fill(mask, -1e9)
        attn = F.softmax(scores / self.tau, dim=-1)
        H_attended = torch.matmul(attn, V)
        H_updated = H + H_attended
        return H_updated, attn
        
    def forward(self, G, tab, seq):
        g_emb = self.g_enc(G.x, G.edge_index, G.edge_attr, G.batch)
        b_emb = self.b_enc.encode(tab)
        t_emb = self.t_enc(seq, return_embedding=True)
        h_g, h_b, h_t = self.proj_g(g_emb), self.proj_b(b_emb), self.proj_t(t_emb)
        gates, uncertainties = self.compute_gates(h_g, h_b, h_t)
        H = torch.stack([h_g, h_b, h_t], dim=1)
        H_updated, attn_weights = self.gated_cross_attention(H, gates)
        fusion_weights = F.softmax(gates, dim=1)
        z_fused = (fusion_weights.unsqueeze(-1) * H_updated).sum(dim=1)
        logit = self.fusion_head(z_fused).squeeze(-1)
        aux_g = self.aux_g(H_updated[:, 0]).squeeze(-1)
        aux_b = self.aux_b(H_updated[:, 1]).squeeze(-1)
        aux_t = self.aux_t(H_updated[:, 2]).squeeze(-1)
        return {
            'logits': logit,
            'aux_logits': torch.stack([aux_g, aux_b, aux_t], dim=1),
            'uncertainties': uncertainties,
            'gates': gates,
            'fusion_weights': fusion_weights,
            'attn_weights': attn_weights,
            'z_fused': z_fused
        }

def gcva_loss(outputs, y, criterion, mu_aux=0.3, mu_unc=0.1, mu_gate=0.05):
    L_main = criterion(outputs['logits'], y)
    L_aux = criterion(outputs['aux_logits'][:, 0], y) + criterion(outputs['aux_logits'][:, 1], y) + criterion(outputs['aux_logits'][:, 2], y)
    L_aux /= 3.0
    aux_probs = torch.sigmoid(outputs['aux_logits'])
    errors = (aux_probs - y.unsqueeze(1)).pow(2)
    L_unc = (outputs['uncertainties'] - errors).abs().mean()
    gates = outputs['gates']
    L_gate = (gates * (1 - gates)).mean()
    L_total = L_main + mu_aux * L_aux + mu_unc * L_unc + mu_gate * L_gate
    return L_total, L_main.item(), L_aux.item(), L_unc.item(), L_gate.item()

def save_conf_roc(y_true, y_prob, thr, out_dir, tag):
    y_true = np.asarray(y_true, int); y_prob = np.asarray(y_prob, float)
    y_pred = (y_prob > thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.imshow(cm, cmap='Blues')
    ax.set_title(f'Confusion ({tag})')
    ax.set_xlabel('Pred'); ax.set_ylabel('True')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Non-Rep','Rep']); ax.set_yticklabels(['Non-Rep','Rep'])
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha='center', va='center')
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(); ensure_dir(out_dir)
    fig.savefig(os.path.join(out_dir, f'cm_{tag}.png')); plt.close(fig)
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob); roc_auc = auc(fpr, tpr)
        plt.figure(); plt.plot(fpr,tpr,label=f"AUC={roc_auc:.3f}"); plt.plot([0,1],[0,1],'--'); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'roc_{tag}.png')); plt.close()
    except Exception:
        roc_auc = float('nan')
    rep = classification_report(y_true, y_pred, target_names=['Non-Rep','Rep'], digits=4)
    with open(os.path.join(out_dir, f"classification_report_{tag}.txt"), "w") as f:
        f.write(rep)
    return roc_auc, rep

def save_training_curves(train_losses, val_losses, val_f1s, out_dir):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('Training and Validation Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, val_f1s, 'g-', label='Val F1')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('F1 Score'); ax2.set_title('Validation F1 Score')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'training_curves.png'), dpi=150); plt.close()

# --- NUEVAS FUNCIONES DE PLOTEO ---
def plot_fusion_weights_violin(all_weights, output_dir, view_names=['Graph', 'Tabular', 'Temporal']):
    """Opción A: Genera Violin Plot de los pesos de fusión."""
    df_w = pd.DataFrame(all_weights, columns=view_names)
    df_melt = df_w.melt(var_name='View', value_name='Attention Weight')
    
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(8, 6))
    ax = sns.violinplot(x="View", y="Attention Weight", data=df_melt, 
                        palette=["#e74c3c", "#3498db", "#2ecc71"],
                        cut=0, inner="box", alpha=0.8)
    plt.title("Distribution of Fusion Weights (Test Set)", fontsize=14, fontweight='bold')
    plt.ylabel("Fusion Weight ($\omega$)", fontsize=12)
    plt.xlabel("")
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fusion_weights_violinplot.png"), dpi=300)
    plt.close()
    print("   ✓ Violin Plot de pesos de fusión guardado.")

def plot_fusion_weights_ternary(all_weights, output_dir, view_names=['Graph', 'Tabular', 'Temporal']):
    ensure_dir(output_dir)
    if all_weights.shape[1] != 3:
        print("   ⚠ El Ternary Plot solo es para 3 vistas. Saltando.")
        return
    
    df_w = pd.DataFrame(all_weights, columns=view_names)
    
    # Crear figura básica
    fig = px.scatter_ternary(df_w, a=view_names[0], b=view_names[1], c=view_names[2],
                            color_discrete_sequence=["#2ecc71"], 
                            title="Ternary Plot of Fusion Weights (Test Set)",
                            opacity=0.6)
    
    # --- CORRECCIÓN DE LA API DE PLOTLY ---
    fig.update_layout(
        ternary={
            'sum': 1,
            'aaxis': {'title': view_names[0], 'min': 0.01, 'linewidth':2, 'ticks':'outside'},
            'baxis': {'title': view_names[1], 'min': 0.01, 'linewidth':2, 'ticks':'outside'},
            'caxis': {'title': view_names[2], 'min': 0.01, 'linewidth':2, 'ticks':'outside'}
        },
        font=dict(size=14),
        margin=dict(t=80, b=40, l=40, r=40)
    )
    # --------------------------------------
    
    try:
        pio.write_image(fig, os.path.join(output_dir, "fusion_weights_ternaryplot.png"), scale=3)
        print("   ✓ Ternary Plot guardado (PNG).")
    except Exception as e:
        print(f"   ⚠ Fallo al guardar PNG estático: {e}. Guardando HTML...")
        fig.write_html(os.path.join(output_dir, "fusion_weights_ternaryplot.html"))
        print("   ✓ Ternary Plot guardado (HTML).")


def main():
    print("\n" + "="*80)
    print("INICIANDO ENTRENAMIENTO GCVA")
    print("="*80 + "\n")
    
    print("[1/10] Cargando configuración...")
    cfg = read_yaml("config/config_three_gcva.yaml")
    if 'tabular_features' not in cfg and 'tab_features' in cfg:
        cfg['tabular_features'] = cfg.pop('tab_features')
    require_keys(cfg, ['seed','csv_paths','target','tabular_features','gnn_model_params','tab_model_params','temporal_params','fusion_params','training_params','output_dir'], 'config')
    require_keys(cfg['csv_paths'], ['graphs','tabular','temporal'], 'csv_paths')
    require_keys(cfg['gnn_model_params'], ['hidden_dim'], 'gnn_model_params')
    require_keys(cfg['tab_model_params'], ['hidden_dim'], 'tab_model_params')
    require_keys(cfg['temporal_params'], ['window'], 'temporal_params')
    require_keys(cfg['training_params'], ['epochs','learning_rate','batch_size','patience'], 'training_params')
    print("   ✓ Configuración cargada correctamente")

    seed = int(cfg['seed']); set_deterministic_seed(seed)
    ensure_dir(cfg['output_dir'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   ✓ Device: {device}")
    print(f"   ✓ Seed: {seed}")

    print("\n[2/10] Cargando datasets...")
    ds = ThreeViewsDataset(
        graphs_csv=cfg['csv_paths']['graphs'],
        tabular_csv=cfg['csv_paths']['tabular'],
        temporal_csv=cfg['csv_paths']['temporal'],
        target_col=cfg['target'],
        tab_features=cfg['tabular_features'],
        window=int(cfg['temporal_params']['window'])
    )
    print(f"   ✓ Total muestras: {len(ds)}")
    print(f"   ✓ Nodos únicos: {len(ds.node_mapping)}")
    print(f"   ✓ Features tabulares: {len(cfg['tabular_features'])}")

    print("\n[3/10] Generando splits por vehículo...")
    plates = ds.df['num_plate'].tolist()
    uniq = np.array(sorted(set(plates)))
    rng = np.random.RandomState(seed); rng.shuffle(uniq)
    n = len(uniq); test_size=float(cfg['training_params'].get('test_size',0.2)); val_size=float(cfg['training_params'].get('val_size',0.1))
    n_test = int(round(test_size*n)); n_val = int(round(val_size*(n-n_test)))
    test_ids = set(uniq[:n_test]); val_ids = set(uniq[n_test:n_test+n_val]); train_ids = set(uniq[n_test+n_val:])
    idx_all = np.arange(len(ds))
    train_idx = [i for i in idx_all if plates[i] in train_ids]
    val_idx   = [i for i in idx_all if plates[i] in val_ids]
    test_idx  = [i for i in idx_all if plates[i] in test_ids]
    if len(train_idx)==0 or len(val_idx)==0 or len(test_idx)==0:
        raise ValueError("Particiones vacías")
    print(f"   ✓ Train: {len(train_idx)} muestras ({len(train_ids)} vehículos)")
    print(f"   ✓ Val:   {len(val_idx)} muestras ({len(val_ids)} vehículos)")
    print(f"   ✓ Test:  {len(test_idx)} muestras ({len(test_ids)} vehículos)")

    print("\n[4/10] Normalizando features tabulares...")
    cols = list(cfg['tabular_features'])
    m = ds.df.loc[train_idx, cols].mean()
    s = ds.df.loc[train_idx, cols].std().replace(0, 1e-6)
    ds.df.loc[:, cols] = (ds.df[cols] - m) / s
    ds.df.loc[:, cols] = ds.df[cols].fillna(0.0)
    print("   ✓ Normalización completada")

    print("\n[5/10] Creando dataloaders...")
    train_set = torch.utils.data.Subset(ds, train_idx)
    val_set   = torch.utils.data.Subset(ds, val_idx)
    test_set  = torch.utils.data.Subset(ds, test_idx)
    g = torch.Generator(); g.manual_seed(seed)
    batch = int(cfg['training_params']['batch_size'])
    train_loader = GeoDataLoader(train_set, batch_size=batch, shuffle=True, generator=g, collate_fn=collate_three)
    val_loader   = GeoDataLoader(val_set,   batch_size=batch, shuffle=False, collate_fn=collate_three)
    test_loader  = GeoDataLoader(test_set,  batch_size=batch, shuffle=False, collate_fn=collate_three)
    print(f"   ✓ Train batches: {len(train_loader)}")
    print(f"   ✓ Val batches:   {len(val_loader)}")
    print(f"   ✓ Test batches:  {len(test_loader)}")

    print("\n[6/10] Construyendo modelo GCVA...")
    node_feat_dim = len(ds.node_mapping)+1
    edge_feat_dim = 1
    tab_in = len(cols)
    model = GCVA(node_feat_dim, edge_feat_dim, tab_in, cfg['gnn_model_params'], cfg['tab_model_params'], cfg['fusion_params']).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✓ Parámetros totales: {total_params:,}")
    print(f"   ✓ Parámetros entrenables: {trainable_params:,}")

    print("\n[7/10] Configurando optimización...")
    y_train = np.array([ds[i][3].item() for i in train_idx], dtype=int)
    pos_w = compute_pos_weight_from_labels(y_train)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['training_params']['learning_rate']))
    print(f"   ✓ Pos weight: {pos_w:.4f}")
    print(f"   ✓ Learning rate: {cfg['training_params']['learning_rate']}")

    total_epochs = int(cfg['training_params']['epochs'])
    patience = int(cfg['training_params']['patience'])
    
    mu_aux = float(cfg['fusion_params'].get('mu_aux', 0.3))
    mu_unc = float(cfg['fusion_params'].get('mu_unc', 0.1))
    mu_gate = float(cfg['fusion_params'].get('mu_gate', 0.05))

    best_val = float("inf"); best_path = os.path.join(cfg['output_dir'], "best_model.pth"); trigger=0
    
    train_losses, val_losses, val_f1s = [], [], []
    
    print("\n[8/10] COMENZANDO ENTRENAMIENTO")
    print("="*80)
    start=time.time()

    for epoch in range(1, total_epochs+1):
        # TRAIN
        model.train(); total=0.0
        for batch_idx, (G, tab, seq, y) in enumerate(train_loader):
            G, tab, seq, y = G.to(device), tab.to(device), seq.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(G, tab, seq)
            loss, main_l, aux_l, unc_l, gate_l = gcva_loss(out, y, criterion, mu_aux, mu_unc, mu_gate)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.item())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"   Epoch {epoch:02d} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}", flush=True)
        
        tr_loss = total/max(1,len(train_loader))
        train_losses.append(tr_loss)

        # VALIDATION
        model.eval()
        v_total, v_probs, v_labels = 0.0, [], []
        weights_sum = torch.zeros(3, dtype=torch.float64)
        n_cnt = 0

        with torch.no_grad():
            for G, tab, seq, y in val_loader:
                G, tab, seq, y = G.to(device), tab.to(device), seq.to(device), y.to(device)
                out = model(G, tab, seq)

                loss, _, _, _, _ = gcva_loss(out, y, criterion, mu_aux, mu_unc, mu_gate)
                v_total += float(loss.item())

                p = torch.sigmoid(out["logits"].view(-1))
                v_probs.extend(p.cpu().numpy().tolist())
                v_labels.extend(y.view(-1).cpu().numpy().tolist())

                weights = out["fusion_weights"]
                weights_sum += weights.sum(dim=0).double().cpu()
                n_cnt += weights.size(0)

        val_loss = v_total / max(1, len(val_loader))
        val_losses.append(val_loss)
        
        thr = best_threshold_from_probs(np.array(v_probs), np.array(v_labels))
        f1  = f1_at_threshold(v_probs, v_labels, thr)
        val_f1s.append(f1)

        weights_arr = (weights_sum / max(1, n_cnt)).numpy()
        weights_str = "[" + ", ".join(f"{w:.3f}" for w in weights_arr) + "]"

        print(
            f"\n>>> Epoch {epoch:02d}/{total_epochs} | "
            f"Train: {tr_loss:.4f} | Val: {val_loss:.4f} | "
            f"F1: {f1:.4f} | Thr: {thr:.3f} | W[G,B,T]={weights_str}",
            flush=True
        )

        if val_loss < best_val:
            best_val = val_loss
            trigger = 0
            torch.save(model.state_dict(), best_path)
            print(f"    ✓ Nuevo mejor modelo! (val_loss: {best_val:.4f})", flush=True)
        else:
            trigger += 1
            print(f"    ⚠ No mejora: {trigger}/{patience}", flush=True)
            if trigger >= patience:
                print(f"\n>>> EARLY STOPPING en epoch {epoch}", flush=True)
                break

    train_time = time.time()-start
    print(f"\n[9/10] Entrenamiento completado en {train_time:.2f}s ({train_time/60:.2f} min)")

    save_training_curves(train_losses, val_losses, val_f1s, cfg['output_dir'])
    print(f"   ✓ Curvas de entrenamiento guardadas")

    print("\n[10/10] Evaluación en test set y Generación de Plots...")
    try:
        state = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
    except TypeError:
        model.load_state_dict(torch.load(best_path, map_location=device))

    # Buscar umbral óptimo en validación
    model.eval()
    v_probs, v_labels = [], []
    with torch.no_grad():
        for G, tab, seq, y in val_loader:
            G, tab, seq, y = G.to(device), tab.to(device), seq.to(device), y.to(device)
            out = model(G, tab, seq)
            v_probs.extend(torch.sigmoid(out["logits"].view(-1)).cpu().numpy().tolist())
            v_labels.extend(y.view(-1).cpu().numpy().tolist())
    final_thr = best_threshold_from_probs(np.array(v_probs), np.array(v_labels))
    print(f"   ✓ Mejor umbral (val): {final_thr:.3f}")

    # Inferencia en TEST + Captura de PESOS
    probs_t, labels_t = [], []
    test_weights_list = [] # Lista para guardar los pesos
    
    with torch.no_grad():
        for G, tab, seq, y in test_loader:
            G, tab, seq, y = G.to(device), tab.to(device), seq.to(device), y.to(device)
            out = model(G, tab, seq)
            probs_t.extend(torch.sigmoid(out["logits"].view(-1)).cpu().numpy().tolist())
            labels_t.extend(y.view(-1).cpu().numpy().tolist())
            # Guardar pesos del batch
            test_weights_list.append(out["fusion_weights"].cpu().numpy())
    
    # Concatenar pesos de todos los batches
    all_test_weights = np.concatenate(test_weights_list, axis=0)
    
    # Generar y guardar los nuevos plots (Violin y Ternary)
    print("   Generando gráficos de pesos de fusión...")
    plot_fusion_weights_violin(all_test_weights, cfg['output_dir'])
    plot_fusion_weights_ternary(all_test_weights, cfg['output_dir'])

    roc_auc, rep_test = save_conf_roc(labels_t, probs_t, final_thr, cfg['output_dir'], "gcva")
    print(f"   ✓ AUC: {roc_auc:.4f}")
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT (TEST):")
    print("="*80)
    print(rep_test)

    with open(os.path.join(cfg['output_dir'], "summary.txt"), "w") as f:
        f.write("=== Gated Cross-View Attention (GCVA) ===\n")
        f.write(f"Tiempo entrenamiento (s): {train_time:.2f}\n")
        f.write(f"Parametros: {total_params:,}\n")
        f.write(f"Umbral (val): {final_thr:.3f}\n")
        f.write(f"AUC (test): {roc_auc:.4f}\n")
        f.write("\nClassification Report (test):\n")
        f.write(rep_test)
    
    print(f"\n✓ Resultados guardados en: {cfg['output_dir']}")
    print("✓ PROCESO COMPLETADO EXITOSAMENTE\n")

if __name__ == "__main__":
    main()