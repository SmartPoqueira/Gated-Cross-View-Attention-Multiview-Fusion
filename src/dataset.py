"""
dataset.py
----------
Dataset and data-loading utilities for the three-view GCVA pipeline.

The three views are:
  - Graph view   : vehicle route as a directed graph (EGAT encoder)
  - Tabular view : IoT tabular features (TabularTransformer encoder)
  - Temporal view: daily traffic counts as a time series (BiLSTM encoder)
"""

import ast
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch


class ThreeViewsDataset(Dataset):
    """
    Multi-view dataset combining graph-structured routes, tabular features,
    and a temporal traffic count sequence.

    Args:
        graphs_csv (str): Path to CSV with columns 'num_plate', 'entry_date',
            'route', 'times', 'directions', and the target column.
        tabular_csv (str): Path to CSV with columns 'num_plate' and tabular features.
        temporal_csv (str): Path to CSV with columns 'day' and 'count'.
        target_col (str): Name of the binary target column.
        tab_features (list[str]): Tabular feature column names.
        window (int): Number of past days for the temporal sequence.
    """

    def __init__(
        self,
        graphs_csv: str,
        tabular_csv: str,
        temporal_csv: str,
        target_col: str,
        tab_features,
        window: int = 7,
    ) -> None:
        gdf = pd.read_csv(graphs_csv, low_memory=False)
        tdf = pd.read_csv(tabular_csv, low_memory=False)
        tmp = pd.read_csv(temporal_csv, low_memory=False)

        if "entry_date" not in gdf.columns:
            raise KeyError("graphs_csv must contain 'entry_date'")

        gdf["entry_date"] = pd.to_datetime(gdf["entry_date"], errors="coerce")

        for col in ("num_plate",):
            if col not in gdf.columns or col not in tdf.columns:
                raise KeyError(f"Both CSVs must contain '{col}'")

        gdf["num_plate"] = gdf["num_plate"].astype(str).str.strip().str.lower()
        tdf["num_plate"] = tdf["num_plate"].astype(str).str.strip().str.lower()

        if target_col not in gdf.columns:
            if "num_visits" in gdf.columns:
                gdf[target_col] = (gdf["num_visits"] >= 2).astype(int)
            else:
                raise KeyError(f"Column '{target_col}' not found and 'num_visits' unavailable.")

        gdf[target_col] = gdf[target_col].astype(int)
        merged = pd.merge(gdf, tdf, on="num_plate", how="inner", suffixes=("", "_tab"))
        if merged.empty:
            raise ValueError("Inner join produced an empty DataFrame.")

        self.tab_features = list(tab_features)
        for c in self.tab_features:
            if c not in merged.columns:
                raise KeyError(f"Tabular feature '{c}' not found in merged DataFrame.")
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

        if "day" not in tmp.columns or "count" not in tmp.columns:
            raise KeyError("temporal_csv must contain 'day' and 'count'.")

        tmp["day"] = pd.to_datetime(tmp["day"])
        tmp["count"] = pd.to_numeric(tmp["count"], errors="coerce")
        tmp["count"] = (tmp["count"] - tmp["count"].mean()) / (tmp["count"].std() + 1e-6)
        self.temporal = tmp.set_index("day")["count"].sort_index()

        self.df = merged.reset_index(drop=True)
        self.target_col = target_col
        self.window = int(window)

        # Build node index mapping from all routes in the dataset
        node_set: set = set()
        for route_str in self.df["route"]:
            for n in ast.literal_eval(route_str):
                node_set.add(n)
        self.node_mapping = {n: i for i, n in enumerate(sorted(node_set))}

    def __len__(self) -> int:
        return len(self.df)

    def _seq_from_temporal(self, day) -> torch.Tensor:
        """Return a window-length daily-count sequence ending on *day*."""
        days = [
            pd.to_datetime(day).normalize() - timedelta(days=i)
            for i in reversed(range(self.window))
        ]
        return torch.tensor(
            [float(self.temporal.get(d, 0.0)) for d in days], dtype=torch.float32
        )

    def _row_to_graph(self, row) -> Data:
        """Convert a single visit record to a PyG Data object."""
        route = ast.literal_eval(row["route"])
        times = ast.literal_eval(row["times"])
        directions = ast.literal_eval(row["directions"])
        n = len(route)

        # Guard against length mismatches introduced by data inconsistencies
        times = times[: max(0, n - 1)]
        directions = (directions + [0] * n)[:n]

        ei = [[i, i + 1] for i in range(n - 1)]
        if ei:
            t = np.asarray(times, dtype=np.float32)
            if t.size:
                t = (t - t.mean()) / (t.std() + 1e-6)
            edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(t[:, None], dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float32)

        # One-hot node features + direction
        X = torch.zeros((n, len(self.node_mapping)), dtype=torch.float32)
        for i, node in enumerate(route):
            X[i, self.node_mapping[node]] = 1.0
        dir_feat = torch.tensor(directions[:n], dtype=torch.float32).unsqueeze(-1)
        X = torch.cat([X, dir_feat], dim=1)

        y = torch.tensor([row[self.target_col]], dtype=torch.float32)
        return Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=y)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        G = self._row_to_graph(row)
        tab_vals = torch.tensor(
            row[self.tab_features].values.astype(np.float32), dtype=torch.float32
        )
        seq = self._seq_from_temporal(row["entry_date"])
        y = torch.tensor(row[self.target_col], dtype=torch.float32)
        return G, tab_vals, seq, y


def collate_three(batch):
    """Collate function for ThreeViewsDataset batches."""
    graphs, tabs, seqs, labels = zip(*batch)
    return (
        Batch.from_data_list(graphs),
        torch.stack(tabs, 0),
        torch.stack(seqs, 0),
        torch.stack(labels, 0).view(-1),
    )
