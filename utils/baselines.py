from torch import nn
from utils.basic_modules import GIN, MLP
from torch_geometric import nn as gnn
from utils.sinkhorn_graph import SinkhornGAT


class GIN_Baseline(nn.Module):
    def __init__(self, data_module, n_hidden_channels, n_graph_layers, n_graph_dropout, n_linear_layers, n_linear_dropout):
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

class Sinkhorn_Baseline(nn.Module):
    def __init__(self, data_module, n_hidden_channels, n_graph_layers, n_graph_dropout, n_linear_layers, n_linear_dropout):
        super(Sinkhorn_Baseline, self).__init__()
        self.sinkhorn_gat = SinkhornGAT(data_module.num_node_features, n_hidden_channels, n_graph_layers, dropout=n_graph_dropout)
        
        self.head = MLP(n_linear_layers, n_hidden_channels, n_hidden_channels, data_module.num_classes, n_linear_dropout, nn.ReLU())

    def forward(self, x, edge_index, batch_sample_indices):
        # 1. Obtain node embeddings 
        x = self.sinkhorn_gat(x, edge_index)

        # 2. Readout layer
        x = gnn.global_mean_pool(x, batch_sample_indices)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x =  self.head(x)
        
        return x