
"""
Baselines
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "21-08-2021"

from torch.utils.tensorboard.writer import SummaryWriter
from utils.one_hot_graph import AttentionOneHotConv, IsomporphismOneHotConv, OneHotGraph
from torch import nn
from utils.basic_modules import GIN, MLP, GAT, GINGAT, AbstractBaseline
from torch_geometric import nn as gnn
from utils.sinkhorn_graph import SinkhornGAT



class GIN_Baseline(AbstractBaseline):
    def __init__(self, data_module, n_hidden_channels, n_graph_layers, n_graph_dropout, n_linear_layers, n_linear_dropout, **kwargs):
        super(GIN_Baseline, self).__init__()
        self.gin = GIN(data_module.num_node_features, n_hidden_channels, n_graph_layers, dropout=n_graph_dropout, **kwargs)
        
        self.head = MLP(n_linear_layers, n_hidden_channels, n_hidden_channels, data_module.num_classes, n_linear_dropout, nn.ReLU())

    def forward(self, x, edge_index, batch_sample_indices, **kwargs):
        # 1. Obtain node embeddings 
        x = self.gin(x, edge_index)

        # 2. Readout layer
        x = gnn.global_mean_pool(x, batch_sample_indices)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x =  self.head(x)
        
        return x

class GAT_Baseline(AbstractBaseline):
    def __init__(self, data_module, n_hidden_channels, n_graph_layers, n_graph_dropout, n_linear_layers, n_linear_dropout, **kwargs):
        super(GAT_Baseline, self).__init__()
        self.gat = GAT(data_module.num_node_features, n_hidden_channels, n_graph_layers, dropout=n_graph_dropout, **kwargs)
        
        self.head = MLP(n_linear_layers, n_hidden_channels, n_hidden_channels, data_module.num_classes, n_linear_dropout, nn.ReLU())

    def forward(self, x, edge_index, batch_sample_indices, **kwargs):
        # 1. Obtain node embeddings 
        x = self.gat(x, edge_index)

        # 2. Readout layer
        x = gnn.global_mean_pool(x, batch_sample_indices)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x =  self.head(x)
        
        return x

class Sinkhorn_Baseline(AbstractBaseline):
    def __init__(self, data_module, n_hidden_channels, n_graph_layers, n_graph_dropout, n_linear_layers, n_linear_dropout, **kwargs):
        super(Sinkhorn_Baseline, self).__init__()
        self.sinkhorn_gat = SinkhornGAT(data_module.num_node_features, n_hidden_channels, n_graph_layers, dropout=n_graph_dropout, **kwargs)
        
        self.head = MLP(n_linear_layers, n_hidden_channels, n_hidden_channels, data_module.num_classes, n_linear_dropout, nn.ReLU(), **kwargs)

    def forward(self, x, edge_index, batch_sample_indices, **kwargs):
        # 1. Obtain node embeddings 
        x = self.sinkhorn_gat(x, edge_index)

        # 2. Readout layer
        x = gnn.global_mean_pool(x, batch_sample_indices)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x =  self.head(x)
        
        return x

class GINGAT_Baseline(AbstractBaseline):
    def __init__(self, data_module, n_hidden_channels, n_graph_layers, n_graph_dropout, n_linear_layers, n_linear_dropout, **kwargs):
        super(GINGAT_Baseline, self).__init__()
        self.gingat = GINGAT(data_module.num_node_features, n_hidden_channels, n_graph_layers, dropout=n_graph_dropout, **kwargs)
        
        self.head = MLP(n_linear_layers, n_hidden_channels, n_hidden_channels, data_module.num_classes, n_linear_dropout, nn.ReLU())

    def forward(self, x, edge_index, batch_sample_indices, **kwargs):
        # 1. Obtain node embeddings 
        x = self.gingat(x, edge_index)

        # 2. Readout layer
        x = gnn.global_mean_pool(x, batch_sample_indices)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x =  self.head(x)
        
        return x

class AttentionOneHotGraph_Baseline(AbstractBaseline):
    def __init__(self, data_module, n_hidden_channels, n_graph_layers, n_graph_dropout, n_linear_layers, n_linear_dropout, **kwargs):
        super(AttentionOneHotGraph_Baseline, self).__init__()

        self.ohg = OneHotGraph(
            data_module.num_node_features, 
            n_hidden_channels,
            n_graph_layers, 
            dropout=n_graph_dropout, 
            conv_type=AttentionOneHotConv, 
            **kwargs
            )
        
        self.head = MLP(n_linear_layers, n_hidden_channels, n_hidden_channels, data_module.num_classes, n_linear_dropout, nn.ReLU())

    def forward(self, x, edge_index, batch_sample_indices, n_sample_nodes, adjs, **kwargs):
        # 1. Obtain node embeddings 
        x = self.ohg(x, edge_index, batch_sample_indices, n_sample_nodes, adjs)

        # 2. Readout layer
        x = gnn.global_mean_pool(x, batch_sample_indices)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x =  self.head(x)
        
        return x

class IsomorphismOneHotGraph_Baseline(AbstractBaseline):
    def __init__(self, data_module, n_hidden_channels, n_graph_layers, n_graph_dropout, n_linear_layers, n_linear_dropout, logger: SummaryWriter, one_hot_channels = 8, **kwargs):
        super(IsomorphismOneHotGraph_Baseline, self).__init__()

        self.ohg = OneHotGraph(
            data_module.num_node_features, 
            n_hidden_channels, 
            n_graph_layers, 
            dropout=n_graph_dropout, 
            conv_type=IsomporphismOneHotConv, 
            one_hot_channels = one_hot_channels,
            **kwargs
            )
        
        self.one_hot_channels = one_hot_channels
        self.logger = logger
        self.head = MLP(n_linear_layers, n_hidden_channels, n_hidden_channels, data_module.num_classes, n_linear_dropout, nn.ReLU())

    def forward(self, x, edge_index, batch_sample_indices, n_sample_nodes, adjs, **kwargs):
        # 1. Obtain node embeddings 
        x = self.ohg(x, edge_index, batch_sample_indices, n_sample_nodes, adjs)

        # 2. Readout layer
        x = gnn.global_mean_pool(x, batch_sample_indices)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x =  self.head(x)
        
        return x

    def epoch_log(self, epoch = 0):
        for name, param in self.named_parameters():
            if "ohg.convs.0.first_linear.weight" == name:
                val = param[:, -self.one_hot_channels:].detach().abs().mean()
                self.logger.add_scalar("OH-Part", val, global_step=epoch)
                del val
