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
import torch


class GIN_Baseline(AbstractBaseline):
    def __init__(
        self,
        data_module,
        n_hidden_channels,
        n_graph_layers,
        p_graph_dropout,
        n_linear_layers,
        p_linear_dropout,
        **kwargs
    ):
        super(GIN_Baseline, self).__init__(**kwargs)
        self.gin = GIN(
            data_module.num_node_features,
            n_hidden_channels,
            n_graph_layers,
            dropout=p_graph_dropout,
            **kwargs
        )

        self.head = MLP(
            n_layers=n_linear_layers,
            input_dim=n_hidden_channels,
            hidden_dim=n_hidden_channels,
            output_dim=data_module.num_classes,
            dropout=p_linear_dropout,
            output_activation=nn.Sigmoid(),
        )

    def forward(self, x, edge_index, batch_sample_indices, **kwargs):
        # 1. Obtain node embeddings
        x = self.gin(x, edge_index)

        # 2. Readout layer
        x = gnn.global_mean_pool(
            x, batch_sample_indices
        )  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.head(x)

        return x


class GAT_Baseline(AbstractBaseline):
    def __init__(
        self,
        data_module,
        n_hidden_channels,
        n_graph_layers,
        p_graph_dropout,
        n_linear_layers,
        p_linear_dropout,
        **kwargs
    ):
        super(GAT_Baseline, self).__init__(**kwargs)
        self.gat = GAT(
            data_module.num_node_features,
            n_hidden_channels,
            n_graph_layers,
            dropout=p_graph_dropout,
            **kwargs
        )

        self.head = MLP(
            n_layers=n_linear_layers,
            input_dim=n_hidden_channels,
            hidden_dim=n_hidden_channels,
            output_dim=data_module.num_classes,
            dropout=p_linear_dropout,
            output_activation=nn.Sigmoid(),
        )

    def forward(self, x, edge_index, batch_sample_indices, **kwargs):
        # 1. Obtain node embeddings
        x = self.gat(x, edge_index)

        # 2. Readout layer
        x = gnn.global_mean_pool(
            x, batch_sample_indices
        )  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.head(x)

        return x


class Sinkhorn_Baseline(AbstractBaseline):
    def __init__(
        self,
        data_module,
        n_hidden_channels,
        n_graph_layers,
        p_graph_dropout,
        n_linear_layers,
        p_linear_dropout,
        **kwargs
    ):
        super(Sinkhorn_Baseline, self).__init__(**kwargs)
        self.sinkhorn_gat = SinkhornGAT(
            data_module.num_node_features,
            n_hidden_channels,
            n_graph_layers,
            dropout=p_graph_dropout,
            **kwargs
        )

        self.head = MLP(
            n_layers=n_linear_layers,
            input_dim=n_hidden_channels,
            hidden_dim=n_hidden_channels,
            output_dim=data_module.num_classes,
            dropout=p_linear_dropout,
            output_activation=nn.Sigmoid(),
            **kwargs
        )

    def forward(self, x, edge_index, batch_sample_indices, **kwargs):
        # 1. Obtain node embeddings
        x = self.sinkhorn_gat(x, edge_index)

        # 2. Readout layer
        x = gnn.global_mean_pool(
            x, batch_sample_indices
        )  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.head(x)

        return x


class GINGAT_Baseline(AbstractBaseline):
    def __init__(
        self,
        data_module,
        n_hidden_channels,
        n_graph_layers,
        p_graph_dropout,
        n_linear_layers,
        p_linear_dropout,
        **kwargs
    ):
        super(GINGAT_Baseline, self).__init__(**kwargs)
        self.gingat = GINGAT(
            data_module.num_node_features,
            n_hidden_channels,
            n_graph_layers,
            dropout=p_graph_dropout,
            **kwargs
        )

        self.head = MLP(
            n_layers=n_linear_layers,
            input_dim=n_hidden_channels,
            hidden_dim=n_hidden_channels,
            output_dim=data_module.num_classes,
            dropout=p_linear_dropout,
            output_activation=nn.Sigmoid(),
        )

    def forward(self, x, edge_index, batch_sample_indices, **kwargs):
        # 1. Obtain node embeddings
        x = self.gingat(x, edge_index)

        # 2. Readout layer
        x = gnn.global_mean_pool(
            x, batch_sample_indices
        )  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.head(x)

        return x


class AttentionOneHotGraph_Baseline(AbstractBaseline):
    def __init__(
        self,
        data_module,
        n_hidden_channels,
        n_graph_layers,
        p_graph_dropout,
        n_linear_layers,
        p_linear_dropout,
        **kwargs
    ):
        super(AttentionOneHotGraph_Baseline, self).__init__(**kwargs)

        self.ohg = OneHotGraph(
            data_module.num_node_features,
            n_hidden_channels,
            n_graph_layers,
            dropout=p_graph_dropout,
            conv_type=AttentionOneHotConv,
            **kwargs
        )

        self.head = MLP(
            n_layers=n_linear_layers,
            input_dim=n_hidden_channels,
            hidden_dim=n_hidden_channels,
            output_dim=data_module.num_classes,
            dropout=p_linear_dropout,
            output_activation=nn.Sigmoid(),
        )

    def forward(
        self, x, edge_index, batch_sample_indices, n_sample_nodes, adjs, xs, **kwargs
    ):
        # 1. Obtain node embeddings
        xs = self.ohg(x, edge_index, batch_sample_indices, n_sample_nodes, adjs, xs)

        # 2. Readout layer
        x = torch.cat(
            [torch.mean(x, dim=0).unsqueeze(0) for x in xs], dim=0
        )  # = global mean pool -> to minibatched tensor

        # 3. Apply a final classifier
        x = self.head(x)

        return x


class IsomorphismOneHotGraph_Baseline(AbstractBaseline):
    def __init__(
        self,
        data_module,
        n_hidden_channels,
        n_graph_layers,
        p_graph_dropout,
        n_linear_layers,
        p_linear_dropout,
        one_hot_channels=8,
        **kwargs
    ):
        super(IsomorphismOneHotGraph_Baseline, self).__init__(**kwargs)

        self.ohg = OneHotGraph(
            data_module.num_node_features,
            n_hidden_channels,
            n_graph_layers,
            dropout=p_graph_dropout,
            conv_type=IsomporphismOneHotConv,
            one_hot_channels=one_hot_channels,
            **kwargs
        )

        self.one_hot_channels = one_hot_channels
        self.head = MLP(
            n_layers=n_linear_layers,
            input_dim=n_hidden_channels,
            hidden_dim=n_hidden_channels,
            output_dim=data_module.num_classes,
            dropout=p_linear_dropout,
            output_activation=nn.Sigmoid(),
        )

    def forward(
        self, x, edge_index, batch_sample_indices, n_sample_nodes, adjs, xs, **kwargs
    ):
        # 1. Obtain node embeddings
        xs = self.ohg(x, edge_index, batch_sample_indices, n_sample_nodes, adjs, xs)

        # 2. Readout layer
        x = torch.cat(
            [torch.mean(x, dim=0).unsqueeze(0) for x in xs], dim=0
        )  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.head(x)

        return x

    def epoch_log(self, epoch=0):
        if self.logger is not None:
            for name, param in self.named_parameters():
                if "ohg.convs.0.mlp.linear.0.weight" == name:  # TODO CHECK THIS
                    val = param[:, -self.one_hot_channels :].detach().abs().mean()
                    self.logger.add_scalar("OH-Part", val, global_step=epoch)
                    del val
