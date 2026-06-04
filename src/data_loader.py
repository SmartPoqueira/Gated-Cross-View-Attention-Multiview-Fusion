# common_data.py
import os, json, ast, random
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import torch

# ------- Semillas y paths -------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ------- Splits por grupo (matrícula) -------
def group_split_indices(groups, test_size=0.2, val_size=0.1, seed=42):
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    all_idx = np.arange(len(groups))
    train_val_idx, test_idx = next(gss1.split(all_idx, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(gss2.split(train_val_idx, groups=groups[train_val_idx]))
    return train_val_idx[train_idx], train_val_idx[val_idx], test_idx

def compute_pos_weight_from_labels(labels: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    return float(n_neg) / max(1.0, float(n_pos))

# ------- Lectura y etiqueta -------
def load_df(csv_path: str, target_col: str = "repeater"):
    df = pd.read_csv(csv_path, low_memory=False)
    if target_col not in df.columns:
        raise ValueError(f"El archivo {csv_path} no contiene la columna '{target_col}'")
    # castea etiqueta a {0,1}
    df["label"] = df[target_col].apply(lambda x: 1 if str(x).lower() in ("1", "true", "t", "yes", "y") else 0).astype(int)
    # normaliza fechas si existen
    for c in ("entry_date", "exit_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    # grupo por matrícula si existe; si no, usa índices
    if "num_plate" in df.columns:
        groups = df["num_plate"].values
    else:
        groups = np.arange(len(df))
    return df, groups

# ------- Tabular: escalado sin fuga -------
def build_tabular_tensors(df: pd.DataFrame, features: list, train_idx, val_idx, test_idx):
    X = df[features].copy()
    for col in features:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")
        else:
            X[col] = X[col].astype("category").cat.codes

    X = X.astype(np.float32).values
    y = df["label"].values.astype(np.float32)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val   = scaler.transform(X[val_idx])
    X_test  = scaler.transform(X[test_idx])
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y[train_idx], dtype=torch.float32),
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y[val_idx], dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y[test_idx], dtype=torch.float32),
        scaler
    )

# ------- GNN: construcción de grafos PyG -------
def _parse_list_cell(s):
    if pd.isna(s): return []
    if isinstance(s, list): return s
    try:
        return ast.literal_eval(s)
    except Exception:
        try:
            return json.loads(s)
        except Exception:
            return []

def build_graph_list(df: pd.DataFrame):
    """
    Requiere columnas: route (lista), times (lista), directions (lista).
    Crea: lista de (x, edge_index, edge_attr, y) para PyG.
    """
    # mapeo de nodos
    node_set = set()
    for route_str in df["route"]:
        route = _parse_list_cell(route_str)
        for n in route:
            node_set.add(n)
    node_mapping = {node: idx for idx, node in enumerate(sorted(node_set))}
    num_unique_nodes = len(node_mapping)

    import torch
    from torch_geometric.data import Data
    graphs = []

    for _, row in df.iterrows():
        route = _parse_list_cell(row["route"])
        times = _parse_list_cell(row.get("times", []))
        directions = _parse_list_cell(row.get("directions", []))

        if len(route) == 0:
            x = torch.zeros((0, num_unique_nodes + 1), dtype=torch.float32)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr  = torch.empty((0, 1), dtype=torch.float32)
            y = torch.tensor([row["label"]], dtype=torch.float32)
            graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
            continue

        # ajusta longitudes si vienen desalineadas
        if len(times) != max(0, len(route)-1):
            times = times[:max(0, len(route)-1)]
        if len(directions) != len(route):
            directions = directions[:len(route)] + [0]*(len(route)-len(directions))

        # edge_index y edge_attr (z-score por visita)
        edges = [[i, i+1] for i in range(max(0, len(route)-1))]
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            t = np.asarray(times, dtype=np.float32)
            if t.size:
                t = (t - t.mean()) / (t.std() + 1e-6)
            edge_attr = torch.tensor(t[:, None], dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr  = torch.empty((0, 1), dtype=torch.float32)

        # nodos: one-hot de tipo + dirección
        node_type = torch.zeros((len(route), num_unique_nodes), dtype=torch.float32)
        dir_feat  = torch.zeros((len(route), 1), dtype=torch.float32)
        for i, node in enumerate(route):
            if node in node_mapping:
                node_type[i, node_mapping[node]] = 1.0
            dir_feat[i, 0] = float(directions[i]) if i < len(directions) else 0.0

        x = torch.cat([node_type, dir_feat], dim=1)
        y = torch.tensor([row["label"]], dtype=torch.float32)
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))

    num_node_features = num_unique_nodes + 1
    num_edge_features = 1
    return graphs, num_node_features, num_edge_features

