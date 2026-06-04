import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
import torch_geometric.utils

# --- Módulo de EGAT para actualización de nodos (versión de cabeza única) ---
class EGATConvSingleHead(MessagePassing):
    def __init__(self, in_channels, out_channels, num_edge_features, dropout=0.6, negative_slope=0.2):
        super(EGATConvSingleHead, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.negative_slope = negative_slope

        # Dividir la salida en dos partes: para nodos (FH) y para aristas (FE)
        self.FH = out_channels // 2       # por ejemplo, si out_channels=128, FH=64
        self.FE = out_channels - self.FH    # en ese caso, FE=64

        self.node_lin = nn.Linear(in_channels, self.FH, bias=False)
        self.edge_lin = nn.Linear(num_edge_features, self.FE, bias=False)

        # Parámetro de atención: vector de dimensión (2*FH + FE)
        self.att = nn.Parameter(torch.Tensor(2 * self.FH + self.FE))
        self.leaky_relu = nn.LeakyReLU(self.negative_slope)

        nn.init.xavier_uniform_(self.node_lin.weight)
        nn.init.xavier_uniform_(self.edge_lin.weight)
        nn.init.xavier_uniform_(self.att.unsqueeze(0))

    def forward(self, x, edge_index, edge_attr):
        N = x.size(0)
        x_proj = self.node_lin(x)  # [N, FH]
        E = edge_attr.size(0)
        edge_proj = self.edge_lin(edge_attr)  # [E, FE]
        out = self.propagate(edge_index, x=x_proj, edge_attr=edge_proj, num_nodes=N, node_dim=0)
        return out  # [N, FH+FE]

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        # x_i, x_j: [E, FH] y edge_attr: [E, FE]
        cat = torch.cat([x_i, x_j, edge_attr], dim=-1)  # [E, 2*FH + FE]
        alpha = self.leaky_relu((cat * self.att).sum(dim=-1))  # [E]
        alpha = torch_geometric.utils.softmax(alpha, index)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training).unsqueeze(-1)
        message = torch.cat([x_j, edge_attr], dim=-1)  # [E, FH+FE]
        return message * alpha  # [E, FH+FE]

    def update(self, aggr_out):
        return aggr_out

# --- Capa EGAT iterativa (actualiza nodos y aristas) ---
class EGATLayer(nn.Module):
    def __init__(self, in_node_dim, in_edge_dim, out_dim, dropout_rate=0.6):
        """
        Se asume que la salida de la capa tendrá dimensión out_dim para nodos
        y que las aristas se actualizarán a la misma dimensión.
        """
        super(EGATLayer, self).__init__()
        self.node_conv = EGATConvSingleHead(in_node_dim, out_dim, in_edge_dim, dropout=dropout_rate)
        # Actualización de aristas mediante MLP: se usan las características del nodo fuente, destino y la arista
        self.edge_update = nn.Sequential(
            nn.Linear(in_node_dim * 2 + in_edge_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr):
        x_new = self.node_conv(x, edge_index, edge_attr)  # [N, out_dim]
        src, dst = edge_index[0], edge_index[1]
        x_src = x[src]   # [E, in_node_dim]
        x_dst = x[dst]   # [E, in_node_dim]
        edge_input = torch.cat([x_src, x_dst, edge_attr], dim=1)  # [E, 2*in_node_dim + in_edge_dim]
        edge_new = self.edge_update(edge_input)  # [E, out_dim]
        return x_new, edge_new

# --- Modelo EGAT con fusión multi-escala ---
class EGAT(nn.Module):
    def __init__(self, num_node_features, num_edge_features, model_params):
        """
        model_params debe incluir:
          - hidden_dim: dimensión interna para nodos (por ejemplo, 128)
          - edge_dim: dimensión interna para aristas (se recomienda igual a hidden_dim)
          - num_layers: número de capas EGAT iterativas (por ejemplo, 2)
          - dropout_rate: tasa de dropout (por ejemplo, 0.6)
        """
        super(EGAT, self).__init__()
        hidden_dim = model_params.get('hidden_dim', 128)
        # Cambiamos el default para edge_dim a hidden_dim (en lugar de hidden_dim//2)
        edge_dim = model_params.get('edge_dim', hidden_dim)
        num_layers = model_params.get('num_layers', 2)
        dropout_rate = model_params.get('dropout_rate', 0.6)

        # Proyección inicial (bottleneck) para nodos y aristas.
        self.node_proj = nn.Linear(num_node_features, hidden_dim)
        self.edge_proj = nn.Linear(num_edge_features, edge_dim)
        
        # Crear las capas EGAT iterativas.
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(EGATLayer(hidden_dim, edge_dim, hidden_dim, dropout_rate))
        
        # Fusión multi-escala: concatenamos la proyección inicial y la salida de cada capa.
        self.merge = nn.Sequential(
            nn.Linear((num_layers + 1) * hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.final_linear = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.elu = nn.ELU()

    def forward(self, x, edge_index, edge_attr, batch):
        # Proyección inicial
        x = self.node_proj(x)  # [N, hidden_dim]
        edge_attr = self.edge_proj(edge_attr)  # [E, edge_dim]
        
        # Almacenamos la proyección inicial (escala 0)
        multi_scale = [x]
        
        # Aplicamos las capas iterativas EGAT
        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr)
            multi_scale.append(x)
        
        # Fusión multi-escala: concatenamos todas las representaciones de nodos
        x_cat = torch.cat(multi_scale, dim=1)  # [N, (num_layers+1)*hidden_dim]
        x_merge = self.merge(x_cat)            # [N, hidden_dim]
        x_merge = self.dropout(self.elu(x_merge))
        
        # Pooling global para obtener la representación del grafo
        x_pool = global_mean_pool(x_merge, batch)
        out = self.final_linear(x_pool)
        return out

