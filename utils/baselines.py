
"""
Baselines
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "21-08-2021"

from utils.one_hot_graph import OneHotGraph
from torch import nn
from utils.basic_modules import GIN, MLP, GAT, GINGAT
from torch_geometric import nn as gnn
from utils.sinkhorn_graph import SinkhornGAT


class GIN_Baseline(nn.Module):
    def __init__(self, data_module, n_hidden_channels, n_graph_layers, n_graph_dropout, n_linear_layers, n_linear_dropout, **kwargs):
        super(GIN_Baseline, self).__init__()
        self.gin = GIN(data_module.num_node_features, n_hidden_channels, n_graph_layers, dropout=n_graph_dropout)
        
        self.head = MLP(n_linear_layers, n_hidden_channels, n_hidden_channels, data_module.num_classes, n_linear_dropout, nn.ReLU())

    def forward(self, x, edge_index, batch_sample_indices):
        # 1. Obtain node embeddings 
        x = self.gin(x, edge_index)

        # 2. Readout layer
        x = gnn.global_mean_pool(x, batch_sample_indices)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x =  self.head(x)
        
        return x

class GAT_Baseline(nn.Module):
    def __init__(self, data_module, n_hidden_channels, n_graph_layers, n_graph_dropout, n_linear_layers, n_linear_dropout, **kwargs):
        super(GAT_Baseline, self).__init__()
        self.gat = GAT(data_module.num_node_features, n_hidden_channels, n_graph_layers, dropout=n_graph_dropout)
        
        self.head = MLP(n_linear_layers, n_hidden_channels, n_hidden_channels, data_module.num_classes, n_linear_dropout, nn.ReLU())

    def forward(self, x, edge_index, batch_sample_indices):
        # 1. Obtain node embeddings 
        x = self.gat(x, edge_index)

        # 2. Readout layer
        x = gnn.global_mean_pool(x, batch_sample_indices)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x =  self.head(x)
        
        return x

class Sinkhorn_Baseline(nn.Module):
    def __init__(self, data_module, n_hidden_channels, n_graph_layers, n_graph_dropout, n_linear_layers, n_linear_dropout, **kwargs):
        super(Sinkhorn_Baseline, self).__init__()
        self.sinkhorn_gat = SinkhornGAT(data_module.num_node_features, n_hidden_channels, n_graph_layers, dropout=n_graph_dropout, **kwargs)
        
        self.head = MLP(n_linear_layers, n_hidden_channels, n_hidden_channels, data_module.num_classes, n_linear_dropout, nn.ReLU(), **kwargs)

    def forward(self, x, edge_index, batch_sample_indices):
        # 1. Obtain node embeddings 
        x = self.sinkhorn_gat(x, edge_index)

        # 2. Readout layer
        x = gnn.global_mean_pool(x, batch_sample_indices)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x =  self.head(x)
        
        return x

class GINGAT_Baseline(nn.Module):
    def __init__(self, data_module, n_hidden_channels, n_graph_layers, n_graph_dropout, n_linear_layers, n_linear_dropout, **kwargs):
        super(GINGAT_Baseline, self).__init__()
        self.gingat = GINGAT(data_module.num_node_features, n_hidden_channels, n_graph_layers, dropout=n_graph_dropout)
        
        self.head = MLP(n_linear_layers, n_hidden_channels, n_hidden_channels, data_module.num_classes, n_linear_dropout, nn.ReLU())

    def forward(self, x, edge_index, batch_sample_indices):
        # 1. Obtain node embeddings 
        x = self.gingat(x, edge_index)

        # 2. Readout layer
        x = gnn.global_mean_pool(x, batch_sample_indices)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x =  self.head(x)
        
        return x

class OneHotGraph_Baseline(nn.Module):
    def __init__(self, data_module, n_hidden_channels, n_graph_layers, n_graph_dropout, n_linear_layers, n_linear_dropout, **kwargs):
        super(OneHotGraph_Baseline, self).__init__()
        self.ohg = OneHotGraph(data_module.num_node_features, n_hidden_channels, n_graph_layers, dropout=n_graph_dropout)
        
        self.head = MLP(n_linear_layers, n_hidden_channels, n_hidden_channels, data_module.num_classes, n_linear_dropout, nn.ReLU())

    def forward(self, x, edge_index, batch_sample_indices):
        # 1. Obtain node embeddings 
        x = self.ohg(x, edge_index)

        # 2. Readout layer
        x = gnn.global_mean_pool(x, batch_sample_indices)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x =  self.head(x)
        
        return x