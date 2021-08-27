"""
Implementation of the OneHotGraph
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "21-08-2021"

from typing import Union, Tuple, Optional, Callable, List
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor


import torch
from torch import nn, Tensor
from torch_geometric import nn as gnn
import torch.nn.functional as F
from torch.nn import Conv1d

from utils.basic_modules import MLP, Symlog

import torch_scatter
import torch_geometric


class AttentionOneHotConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True,
        one_hot_attention: str = "dot",
        one_hot_mode: str = "conv",
        one_hot_incay: str = "add",
        one_hot_channels: int = 8,
        first_n_one_hot: int = 10,
        **kwargs
    ):
        """
        
        Parameters
        ----------
        in_channels: int
            The number of incoming channels for each node
        out_channels: int
            The number of outcoming channels for each node
        heads: int
            The number of attention heads that are being used
        concat: bool
            Whether the attention-heads should be concatenated or averaged (as in Velickovic et al.'s paper) => But they average in output layers!
        negative_slope: float
            The negative slope for the LeakyReLU activation used for the node-attetion
        dropout: float
            The dropout probability for the attention weights
        add_self_loops: bool
            Whether or not to add self loops for each node
        bias: bool
            Whether or not to use bias weights for the final computation of the next hidden states
        one_hot_attention: str
            Which attention function to use between the one hot weights.
            Possible are "dot", "none" and "uoi" (Union over Intersection, which is the inverse of the Jaccard metric or the Intersection over Union)
        one_hot_mode: str
            Which type function to use for computing the one hot channels which are added to the hidden state of the nodes and used for further computations.
            Possible are "conv", "none" and "ffn".
        one_hot_incay: str
            How to increase the one-hot-vector with each message passing.
            Possible are "add" which fully adds, "binary_add" which adds 1 to each index if the neighbor has not zero there
            and "indicator" which does not increase but only indicates with 1 if the node has seen this neighbors information
        one_hot_channels: int
            The number of channels to use for computing the one hot vecor which is added to the hidden vector of each node.
        first_n_one_hot: int
            The number of fist nodes that are used to feed into the ffnn to compute the one-hot vecor which is concatenated to the hidden vector of each node.
            This is only relevant if the one_hot_mode = "ffn"
        """

        super(AttentionOneHotConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.one_hot_attention = one_hot_attention
        self.one_hot_mode = one_hot_mode
        self.one_hot_incay = one_hot_incay
        self.first_n_one_hot = first_n_one_hot

        if one_hot_mode == "none":
            one_hot_channels = 0

        self.one_hot_cannels = one_hot_channels

        self.lin = nn.Linear(
            in_channels + one_hot_channels, heads * out_channels, bias=False
        )

        self.att_l = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = nn.Parameter(torch.Tensor(1, heads, out_channels))

        self.pre_att_act = Symlog()
        self.pre_info_act = Symlog()

        if self.one_hot_mode == "conv":
            self.onehot_pipe = nn.Sequential(  # This is very arbitrary
                nn.Conv1d(1, 8, 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(8, 16, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(16, one_hot_channels),
            )

        if self.one_hot_mode == "ffn":
            self.onehot_pipe = MLP(
                n_layers=2,
                input_dim=first_n_one_hot,
                hidden_dim=64,
                output_dim=one_hot_channels,
                use_batch_norm=True,
            )


        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        gnn.inits.glorot(self.lin.weight)
        gnn.inits.glorot(self.att_l)
        gnn.inits.glorot(self.att_r)
        gnn.inits.zeros(self.bias)

    def index_select_onehots(self, onehots, adjs):

        selected_onehots = []

        for onehot, adj in zip(onehots, adjs):
            selected_onehot = onehot.index_select(0, adj)
            selected_onehots.append(selected_onehot)

        return selected_onehots

    def forward(
        self,
        xs: List[Tensor],
        onehots: List[Tensor],
        adjs: List[Tensor],
        n_sample_nodes: Tensor,
        **kwargs
    ):
        """

        Parameters
        ----------
        x: Union[torch.Tensor, OptPairTensor]
            The main channels of the graph-nodes
        onehots: List[Tensor]
            The 'special' one-hot channels of each node. Each list element corresponds to one graph
        edge_index: Adj
            The sending and receiving graph connection. I.e. the adjacent matrix of all the graphs in one batch
        batch_sample_indices: Tensor
            The index of the sample inside the minibatch the node corresponds to
        n_sample_nodes: Tensor
            For each graph in the minibatch the number of nodes
        adjs: List[Tensor]
            The same as in edge_index but seperate for each graph in the minibatch
        """

        res_xs, res_onehot = [], []
        # Onehot Convolution (mine)
        for x, adj, onehot, n_nodes in zip(xs, adjs, onehots, n_sample_nodes):

            if self.one_hot_mode == "conv":
                prepared_onehot = self.pre_info_act(onehot.sort(dim=-1)[0].unsqueeze(1))

            if self.one_hot_mode == "ffn":
                prepared_onehot = self.pre_info_act(
                    onehot.sort(dim=-1, descending=True)[0]
                )
                if prepared_onehot.shape[1] >= self.first_n_one_hot:
                    prepared_onehot = prepared_onehot[:, : self.first_n_one_hot]
                else:
                    prepared_onehot = torch.hstack(
                        (
                            prepared_onehot,
                            torch.zeros(
                                (
                                    n_nodes,
                                    self.first_n_one_hot - prepared_onehot.shape[1],
                                ),
                                device=prepared_onehot.device,
                            ),
                        )
                    )

            if self.one_hot_mode != "none":
                prepared_onehot = self.onehot_pipe(prepared_onehot)
                x = torch.hstack((x, prepared_onehot))

            # if I only add the onehotvectors here I doubt that one linear transform will be enough.
            x = self.lin(x).view(-1, self.heads, self.out_channels)

            sending_alphas = (x * self.att_l).sum(dim=-1)
            receiving_alphas = (x * self.att_r).sum(dim=-1)

            if self.add_self_loops:

                adj, _ = torch_geometric.utils.remove_self_loops(adj)
                adj, _ = torch_geometric.utils.add_self_loops(adj, num_nodes=n_nodes)

            x, onehots = self.propagate(
                x, onehot, sending_alphas, receiving_alphas, adj, n_nodes
            )

            if self.concat:
                x = x.view(-1, self.heads * self.out_channels)
            else:
                x = x.mean(dim=1)

            if self.bias is not None:
                x += self.bias

            res_xs.append(x)
            res_onehot.append(onehot)

        return res_xs, res_onehot

    def propagate(
        self,
        x: Tensor,
        onehot: Tensor,
        sending_alphas: Tensor,
        receiving_alphas: Tensor,
        adj: Tensor,
        n_nodes: int,
        **kwargs
    ):

        sending_indices = adj[0]
        receiving_indices = adj[1]

        sending_x = x.index_select(0, sending_indices)
        sending_onehots = onehot.index_select(0, sending_indices)
        receiving_onehots = onehot.index_select(0, receiving_indices)
        sending_alphas = sending_alphas.index_select(0, sending_indices)
        receiving_alphas = receiving_alphas.index_select(0, receiving_indices)

        weighted_selected_x, weighted_selected_onethots = self.message(
            sending_x=sending_x,
            sending_onehots=sending_onehots,
            receiving_onehots=receiving_onehots,
            sending_alphas=sending_alphas,
            receiving_alphas=receiving_alphas,
            receiving_indices=receiving_indices,
        )

        aggregated_selected_x, aggregated_selected_onehots = self.aggregate(
            weighted_selected_x, weighted_selected_onethots, receiving_indices, n_nodes
        )

        new_x, new_onehots = self.update(
            aggregated_selected_x, aggregated_selected_onehots, onehot
        )

        return new_x, new_onehots

    def message(
        self,
        sending_x: Tensor,
        sending_onehots: List[Tensor],
        receiving_onehots: List[Tensor],
        sending_alphas: Tensor,
        receiving_alphas: OptTensor,
        receiving_indices: Tensor,
        **kwargs
    ) -> Tuple[Tensor, Tensor]:

        alpha = (
            sending_alphas
            if receiving_alphas is None
            else sending_alphas + receiving_alphas
        )
        alpha = F.leaky_relu(alpha, self.negative_slope)

        if self.one_hot_attention != "none":

            sending = self.pre_att_act(sending_onehots)
            receiving = self.pre_att_act(receiving_onehots)

            if self.one_hot_attention == "dot":
                onehot_dot_attention = 1 / (
                    torch.sum(sending * receiving, dim=1) + 1
                )  # +1 for zeros

            if (
                self.one_hot_attention == "uoi"
                or self.one_hot_attention == "union_over_intersection"
            ):  # =inverse tanimoto=inverse intersection over union = inverse jaccard
                onehot_dot_attention = torch.sum(
                    torch.maximum(sending, receiving), dim=-1
                ) / (
                    torch.sum(torch.minimum(sending, receiving), dim=-1) + 1
                )  # +1 for zeros

            alpha = alpha * onehot_dot_attention.unsqueeze(-1)

        alpha = torch_geometric.utils.softmax(alpha, receiving_indices)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        weighted_selected_x = sending_x * alpha.unsqueeze(-1)

        if self.one_hot_incay == "binary_add":
            sending_onehots = sending_onehots != 0

        return weighted_selected_x, sending_onehots

    def aggregate(
        self,
        sending_x,
        sending_onehots,
        receiving_indices: Tensor,
        n_nodes: Tensor,
        **kwargs
    ) -> Tuple[Tensor, List[Tensor]]:

        aggregated_selected_x = torch_scatter.scatter(
            sending_x, receiving_indices, dim=0, dim_size=n_nodes, reduce="sum"
        )
        aggregated_selected_onehots = torch_scatter.scatter(
            sending_onehots, receiving_indices, dim=0, dim_size=n_nodes, reduce="add"
        )

        return aggregated_selected_x, aggregated_selected_onehots

    def update(
        self, message_x: Tensor, message_onehots: Tensor, onehots: Tensor, **kwargs
    ) -> Tuple[Tensor, List[Tensor]]:

        if self.one_hot_incay == "indicator":
            new_one_hots = (message_onehots + onehots) != 0
        else:
            new_one_hots = message_onehots + onehots

        return message_x, new_one_hots


class IsomporphismOneHotConv(nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        one_hot_mode: str = "conv",
        one_hot_channels: int = 8,
        first_n_one_hot: int = 16,
        **kwargs
    ):

        super(IsomporphismOneHotConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.one_hot_mode = one_hot_mode
        self.one_hot_cannels = one_hot_channels
        self.first_n_one_hot = first_n_one_hot

        self.mlp = MLP(
            n_layers=2,
            input_dim=in_channels + one_hot_channels,
            output_dim=out_channels,
            hidden_dim=out_channels,
            use_batch_norm=True,
            output_activation=None,
        )

        self.info_act = Symlog()

        if self.one_hot_mode == "conv":
            self.onehot_pipe = nn.Sequential(  # This is very arbitrary
                nn.Conv1d(1, 8, 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(8, 16, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(16, one_hot_channels),
            )

        if self.one_hot_mode == "ffn":
            self.onehot_pipe = MLP(
                n_layers=2,
                input_dim=first_n_one_hot,
                hidden_dim=64,
                output_dim=one_hot_channels,
                use_batch_norm=True,
            )

        self.reset_parameters()

    def reset_parameters(self):
        pass  # TODO

    def index_select_onehots(self, onehots, adjs):

        selected_onehots = []

        for onehot, adj in zip(onehots, adjs):
            selected_onehot = onehot.index_select(0, adj[0])
            selected_onehots.append(selected_onehot)

        return selected_onehots

    def forward(
        self,
        xs: List[Tensor],
        onehots: List[Tensor],
        adjs: List[Tensor],
        n_sample_nodes: Tensor,
        **kwargs
    ):
        """

        Parameters
        ----------
        x: Union[torch.Tensor, OptPairTensor]
            The main channels of the graph-nodes
        onehots: List[Tensor]
            The 'special' one-hot channels of each node. Each list element corresponds to one graph
        edge_index: Adj
            The sending and receiving graph connection. I.e. the adjacent matrix of all the graphs in one batch
        batch_sample_indices: Tensor
            The index of the sample inside the minibatch the node corresponds to
        n_sample_nodes: Tensor
            For each graph in the minibatch the number of nodes
        adjs: List[Tensor]
            The same as in edge_index but seperate for each graph in the minibatch
        """

        res_xs, res_onehot = [], []
        # Onehot Convolution (mine)
        for x, adj, onehot, n_nodes in zip(xs, adjs, onehots, n_sample_nodes):

            x, onehot = self.propagate(x, onehot, adj, n_nodes)

            if self.one_hot_mode == "conv":
                # Unsqueezing the channel dimension for convs
                prepared_onehot = self.info_act(onehot.sort(dim=-1)[0].unsqueeze(1))

            if self.one_hot_mode == "ffn":
                prepared_onehot = self.info_act(onehot.sort(dim=-1, descending=True)[0])
                if prepared_onehot.shape[1] >= self.first_n_one_hot:
                    prepared_onehot = prepared_onehot[:, : self.first_n_one_hot]
                else:
                    prepared_onehot = torch.hstack(
                        (
                            prepared_onehot,
                            torch.zeros(
                                (
                                    n_nodes,
                                    self.first_n_one_hot - prepared_onehot.shape[1],
                                ),
                                device=prepared_onehot.device,
                            ),
                        )
                    )

            prepared_onehot = self.onehot_pipe(prepared_onehot)
            x = torch.hstack((x, prepared_onehot))

            x = self.mlp(x)

            res_xs.append(x)
            res_onehot.append(onehot)

        return res_xs, res_onehot

    def propagate(self, x, onehot, adj, n_nodes, **kwargs):

        sending_indices = adj[0]
        receiving_indices = adj[1]

        selected_onehots = onehot.index_select(0, sending_indices)
        selected_x = x.index_select(0, sending_indices)

        weighted_selected_x, weighted_selected_onethots = self.message(
            selected_x=selected_x,
            selected_onehots=selected_onehots,
            n_sample_nodes=n_nodes,
        )

        aggregated_selected_x, aggregated_selected_onehots = self.aggregate(
            weighted_selected_x, weighted_selected_onethots, receiving_indices, n_nodes
        )

        new_x, new_onehots = self.update(
            aggregated_selected_x, aggregated_selected_onehots, onehot
        )

        return new_x, new_onehots

    def message(
        self, selected_x: Tensor, selected_onehots: Tensor, **kwargs
    ) -> Tuple[Tensor, List[Tensor]]:

        return selected_x, selected_onehots

    def aggregate(
        self,
        sending_x,
        sending_onehots,
        receiving_indices: Tensor,
        n_nodes: Tensor,
        **kwargs
    ) -> Tuple[Tensor, List[Tensor]]:

        message_x = torch_scatter.scatter(
            sending_x, receiving_indices, dim=0, dim_size=n_nodes, reduce="sum"
        )
        message_onehots = torch_scatter.scatter(
            sending_onehots, receiving_indices, dim=0, dim_size=n_nodes, reduce="add"
        )

        return message_x, message_onehots

    def update(
        self,
        message_x: Tensor,
        message_onehots: List[Tensor],
        onehots: List[Tensor],
        **kwargs
    ) -> Tuple[Tensor, List[Tensor]]:

        return message_x, onehots + message_onehots


class OneHotGraph(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float = 0.0,
        heads=1,
        act: Optional[Callable] = nn.ReLU(inplace=True),
        conv_type=AttentionOneHotConv,
        **kwargs
    ):
        super(OneHotGraph, self).__init__()

        if conv_type == AttentionOneHotConv:
            assert hidden_channels % heads == 0
            out_channels = hidden_channels // heads
        if conv_type == IsomporphismOneHotConv:
            out_channels = hidden_channels

        self.convs = nn.ModuleList()

        self.convs.append(
            conv_type(in_channels, out_channels, dropout=dropout, heads=heads, **kwargs)
        )
        for _ in range(1, num_layers):
            self.convs.append(
                conv_type(hidden_channels, out_channels, heads=heads, **kwargs)
            )

        self.dropout = dropout
        self.act = act
        self.num_layers = num_layers

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        batch_sample_indices,
        n_sample_nodes,
        adjs,
        xs,
        *args,
        **kwargs
    ) -> Tensor:

        device = x.device
        one_hots = OneHotGraph.initializeOneHots(n_sample_nodes, device)

        for i in range(self.num_layers):
            xs, one_hots = self.convs[i](
                xs, one_hots, adjs, n_sample_nodes, *args, **kwargs
            )
            xs = [self.act(x) for x in xs]
            xs = [F.dropout(x, p=self.dropout, training=self.training) for x in xs]

        return xs

    @staticmethod
    def initializeOneHots(n_sample_nodes, device):

        one_hots = []
        for n in n_sample_nodes:
            one_hots.append(torch.eye(n, device=device))

        return one_hots
