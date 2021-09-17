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

from utils.basic_modules import MLP, Sparse3DMLP, Symlog

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
        bias: bool = True,
        use_normal_attention: bool = True,
        one_hot_attention: str = "dot",
        one_hot_mode: str = "conv",
        one_hot_incay: str = "add",
        one_hot_channels: int = 8,
        first_n_one_hot: int = 10,
        one_hot_att_constant: float = 1.0,
        train_one_hot_att_constant: bool = False,
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
        bias: bool
            Whether or not to use bias weights for the final computation of the next hidden states
        use_normal_attention: bool
            Wether or not to use the "normal" GAT attention mechanism which is dependent on the nodes.
            This makes attention heads unnecessary and thus it is required that heads = 1.
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

        assert use_normal_attention or heads == 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.one_hot_attention = one_hot_attention
        self.one_hot_mode = one_hot_mode
        self.one_hot_incay = one_hot_incay
        self.first_n_one_hot = first_n_one_hot
        self.use_normal_attention = use_normal_attention
        if train_one_hot_att_constant:
            self.one_hot_att_constant = nn.Parameter(torch.tensor(one_hot_att_constant, dtype=torch.float32), requires_grad=True)
        else:
            self.one_hot_att_constant = one_hot_att_constant

        if one_hot_mode == "none":
            one_hot_channels = 0

        self.one_hot_cannels = one_hot_channels

        self.lin = nn.Linear(
            in_channels + one_hot_channels, heads * out_channels, bias=False
        )

        if self.use_normal_attention:
            self.att_l = nn.Parameter(torch.Tensor(1, heads, out_channels))
            self.att_r = nn.Parameter(torch.Tensor(1, heads, out_channels))

        self.pre_att_act = Symlog(inplace=False)
        self.pre_info_act = Symlog(inplace=True)

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
        if self.use_normal_attention:
            gnn.inits.glorot(self.att_l)
            gnn.inits.glorot(self.att_r)
        gnn.inits.zeros(self.bias)

    def forward(self, xs: List[Tensor], onehots: List[Tensor], **kwargs):
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

        if self.one_hot_mode == "conv":
            # Unsqueezing the channel dimension for convs
            prepared_onehots = self.pre_info_act(onehots.sort(dim=-1)[0].unsqueeze(-2))

            oh_shape = prepared_onehots.shape
            prepared_onehots = prepared_onehots.reshape(-1, *oh_shape[-2:])
            prepared_onehots = self.onehot_pipe(prepared_onehots).reshape(
                *oh_shape[:-2], -1
            )

        elif self.one_hot_mode == "ffn":
            prepared_onehots = self.pre_info_act(
                onehots.sort(dim=-1, descending=True)[0]
            )
            if prepared_onehots.shape[-1] >= self.first_n_one_hot:
                prepared_onehots = prepared_onehots[:, :, : self.first_n_one_hot]
            else:
                prepared_onehots = torch.cat(
                    (
                        prepared_onehots,
                        torch.zeros(
                            (
                                xs.shape[0],
                                xs.shape[1],
                                self.first_n_one_hot - prepared_onehots.shape[-1],
                            ),
                            device=prepared_onehots.device,
                        ),
                    ),
                    dim=-1,
                )

            prepared_onehots = self.onehot_pipe(prepared_onehots)

        if self.one_hot_mode != "none":
            xs = torch.cat((xs, prepared_onehots), dim=-1)

        xs = self.lin(xs).view(*xs.shape[:-1], self.heads, self.out_channels)

        if self.use_normal_attention:
            sending_alphas = (xs * self.att_l).sum(dim=-1)
            receiving_alphas = (xs * self.att_r).sum(dim=-1)

        if self.use_normal_attention:
            xs, onehots = self.propagate(
                xs=xs,
                onehots=onehots,
                sending_alphas=sending_alphas,
                receiving_alphas=receiving_alphas,
                **kwargs
            )
        else:
            xs, onehots = self.propagate(xs=xs, onehots=onehots, **kwargs)

        if self.concat:
            xs = xs.view(*xs.shape[:-2], -1)
        else:
            xs = xs.mean(dim=-2)

        if self.bias is not None:
            xs += self.bias

        return xs, onehots

    def propagate(
        self,
        xs: Tensor,
        onehots: Tensor,
        adjs: Tensor,
        sending_alphas: Optional[Tensor] = None,
        receiving_alphas: Optional[Tensor] = None,
        **kwargs
    ):

        sending_indices = (adjs[:, 0]).long()  # is ok
        receiving_indices = (adjs[:, 1]).long()

        sending_onehots = torch.gather(
            onehots,
            dim=-2,
            index=sending_indices.unsqueeze(-1).expand(-1, -1, onehots.shape[-1]),
        )
        receiving_onehots = torch.gather(
            onehots,
            dim=-2,
            index=receiving_indices.unsqueeze(-1).expand(-1, -1, onehots.shape[-1]),
        )
        sending_x = torch.gather(
            xs,
            dim=-3,
            index=sending_indices.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, *xs.shape[-2:]),
        )
        # above is ok
        if sending_alphas is not None:
            # sending_alphas = sending_alphas.index_select(0, sending_indices)
            sending_alphas = torch.gather(
                sending_alphas,
                dim=-2,
                index=sending_indices.unsqueeze(-1).expand(
                    -1, -1, sending_alphas.shape[-1]
                ),
            )
        if receiving_alphas is not None:
            # receiving_alphas = receiving_alphas.index_select(0, receiving_indices)
            receiving_alphas = torch.gather(
                receiving_alphas,
                dim=-2,
                index=receiving_indices.unsqueeze(-1).expand(
                    -1, -1, receiving_alphas.shape[-1]
                ),
            )

        weighted_selected_x, weighted_selected_onethots = self.message(
            sending_x=sending_x,
            sending_onehots=sending_onehots,
            receiving_onehots=receiving_onehots,
            receiving_indices=receiving_indices,
            sending_alphas=sending_alphas,
            receiving_alphas=receiving_alphas,
            **kwargs
        )

        aggregated_selected_x, aggregated_selected_onehots = self.aggregate(
            weighted_selected_x, weighted_selected_onethots, receiving_indices, **kwargs
        )

        new_x, new_onehots = self.update(
            aggregated_selected_x, aggregated_selected_onehots, onehots, **kwargs
        )

        return new_x, new_onehots

    def message(
        self,
        sending_x: Tensor,
        sending_onehots: List[Tensor],
        receiving_onehots: List[Tensor],
        receiving_indices: Tensor,
        dim_size: int,
        sending_alphas: Optional[Tensor] = None,
        receiving_alphas: Optional[Tensor] = None,
        **kwargs
    ) -> Tuple[Tensor, Tensor]:

        if sending_alphas is None:
            alpha = torch.ones_like(receiving_indices, dtype=torch.float32).unsqueeze(
                -1
            )
        else:
            alpha = (
                sending_alphas
                if receiving_alphas is None
                else sending_alphas + receiving_alphas
            )
            alpha = F.leaky_relu(alpha, self.negative_slope)

        if self.one_hot_attention != "none":

            sending = self.pre_att_act(
                sending_onehots
            )  # TODO: Before I accidentially made it inplace (and used the changed one later) but the results were good. Keep this in mind!
            receiving = self.pre_att_act(receiving_onehots)

            if self.one_hot_attention == "dot":
                onehot_dot_attention = 1 / (
                    torch.sum(sending * receiving, dim=-1) + self.one_hot_att_constant
                )

            if (
                self.one_hot_attention == "uoi"
                or self.one_hot_attention == "union_over_intersection"
            ):  # =inverse tanimoto=inverse intersection over union = inverse jaccard
                onehot_dot_attention = torch.sum(
                    torch.maximum(sending, receiving), dim=-1
                ) / (
                    torch.sum(torch.minimum(sending, receiving), dim=-1)
                    + self.one_hot_att_constant
                )  # +1 for zeros

            alpha = alpha * onehot_dot_attention.unsqueeze(
                -1
            )  # TODO: This allows interesting dynamics. Maybe softmax alpha before that...

        # = softmax
        max_alphas = torch_scatter.scatter(
            alpha, receiving_indices, dim=-2, dim_size=(dim_size), reduce="max"
        )
        max_alphas = torch.gather(
            max_alphas,
            dim=-2,
            index=receiving_indices.unsqueeze(-1).expand(-1, -1, max_alphas.shape[-1]),
        )
        alpha = (alpha - max_alphas).exp()

        summed_exp = torch_scatter.scatter(
            alpha, receiving_indices, dim=-2, dim_size=(dim_size), reduce="sum"
        )

        alpha = alpha / torch.gather(
            summed_exp,
            dim=-2,
            index=receiving_indices.unsqueeze(-1).expand(-1, -1, summed_exp.shape[-1]),
        )
        # is ok above

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
        dim_size: int,
        **kwargs
    ) -> Tuple[Tensor, List[Tensor]]:

        message_x = torch_scatter.scatter(
            sending_x, receiving_indices, dim=-3, dim_size=(dim_size), reduce="sum",
        )
        message_onehots = torch_scatter.scatter(
            sending_onehots,
            receiving_indices,
            dim=-2,
            dim_size=(dim_size),
            reduce="add",
        )

        return message_x, message_onehots

    def update(
        self, message_x: Tensor, message_onehots: Tensor, onehots: Tensor, **kwargs
    ) -> Tuple[Tensor, List[Tensor]]:

        if self.one_hot_incay == "indicator":
            new_one_hots = ((message_onehots + onehots) != 0).float()
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
        train_eps: bool = False,
        eps: float = 0,
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

        self.mlp = Sparse3DMLP(
            n_layers=2,
            input_dim=in_channels + one_hot_channels,
            output_dim=out_channels,
            hidden_dim=out_channels,
            use_batch_norm=True,
            output_activation=None,
        )

        self.info_act = Symlog(inplace=True)

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

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))

        self.reset_parameters()

    def reset_parameters(self):
        pass  # TODO

    def forward(self, xs: Tensor, onehots: Tensor, n_nodes: Tensor, **kwargs):
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
        n_nodes: Tensor
            For each graph in the minibatch the number of nodes
        adjs: List[Tensor]
            The same as in edge_index but seperate for each graph in the minibatch
        """

        x_old = xs
        xs, onehots = self.propagate(xs, onehots, **kwargs)

        xs = xs + x_old * (
            1 + self.eps
        )  # this should actually be inside the update function... The pytorch geometric guys must be trippin

        if self.one_hot_mode == "conv":
            # Unsqueezing the channel dimension for convs
            prepared_onehots = self.info_act(onehots.sort(dim=-1)[0].unsqueeze(-2))

            oh_shape = prepared_onehots.shape
            prepared_onehots = prepared_onehots.reshape(-1, *oh_shape[-2:])
            prepared_onehots = self.onehot_pipe(prepared_onehots).reshape(
                *oh_shape[:-2], -1
            )

        elif self.one_hot_mode == "ffn":
            prepared_onehots = self.info_act(onehots.sort(dim=-1, descending=True)[0])
            if prepared_onehots.shape[-1] >= self.first_n_one_hot:
                prepared_onehots = prepared_onehots[:, :, : self.first_n_one_hot]
            else:
                prepared_onehots = torch.cat(
                    (
                        prepared_onehots,
                        torch.zeros(
                            (
                                xs.shape[0],
                                xs.shape[1],
                                self.first_n_one_hot - prepared_onehots.shape[-1],
                            ),
                            device=prepared_onehots.device,
                        ),
                    ),
                    dim=-1,
                )
            prepared_onehots = self.onehot_pipe(prepared_onehots)

        xs = torch.cat((xs, prepared_onehots), dim=-1)

        xs = self.mlp(xs, n_nodes)

        return xs, onehots

    def propagate(self, xs, onehots, adjs, **kwargs):

        sending_indices = (adjs[:, 0]).long()  # might be wrong(gather<-)
        receiving_indices = (adjs[:, 1]).long()

        selected_onehots = torch.gather(
            onehots,
            dim=-2,
            index=sending_indices.unsqueeze(-1).expand(-1, -1, onehots.shape[-1]),
        )
        selected_x = torch.gather(
            xs, dim=-2, index=sending_indices.unsqueeze(-1).expand(-1, -1, xs.shape[-1])
        )

        weighted_selected_x, weighted_selected_onethots = self.message(
            selected_x=selected_x, selected_onehots=selected_onehots, **kwargs
        )

        aggregated_selected_x, aggregated_selected_onehots = self.aggregate(
            weighted_selected_x, weighted_selected_onethots, receiving_indices, **kwargs
        )

        new_x, new_onehots = self.update(
            aggregated_selected_x, aggregated_selected_onehots, onehots, **kwargs
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
        dim_size: int,
        **kwargs
    ) -> Tuple[Tensor, List[Tensor]]:

        message_x = torch_scatter.scatter(
            sending_x, receiving_indices, dim=-2, dim_size=(dim_size), reduce="sum",
        )
        message_onehots = torch_scatter.scatter(
            sending_onehots,
            receiving_indices,
            dim=-2,
            dim_size=(dim_size),
            reduce="add",
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
        use_normal_attention=True,
        act: Optional[Callable] = nn.ReLU(inplace=True),
        conv_type=AttentionOneHotConv,
        **kwargs
    ):
        super(OneHotGraph, self).__init__()

        if conv_type == AttentionOneHotConv:
            assert hidden_channels % heads == 0
            assert heads == 1 or use_normal_attention

            out_channels = hidden_channels // heads
        if conv_type == IsomporphismOneHotConv:
            out_channels = hidden_channels

        self.convs = nn.ModuleList()

        self.convs.append(
            conv_type(
                in_channels,
                out_channels,
                dropout=dropout,
                heads=heads,
                use_normal_attention=use_normal_attention,
                **kwargs
            )
        )
        for _ in range(1, num_layers):
            self.convs.append(
                conv_type(
                    hidden_channels,
                    out_channels,
                    heads=heads,
                    use_normal_attention=use_normal_attention,
                    **kwargs
                )
            )

        self.dropout = dropout
        self.act = act
        self.num_layers = num_layers

    def forward(self, xs: Tensor, onehots: Tensor, **kwargs) -> Tensor:

        for i in range(self.num_layers):
            xs, onehots = self.convs[i](xs, onehots, **kwargs)
            xs = self.act(xs)
            xs = F.dropout(xs, p=self.dropout, training=self.training)

        return xs
