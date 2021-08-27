"""
Utility classes for basic modules to be used as components
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "21-08-2021"

from typing import Optional, Callable, List
from torch_geometric.typing import OptPairTensor, Adj, Size, NoneType


import copy

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU

from torch_geometric.nn.conv import GCNConv, SAGEConv, GINConv, GATConv
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge

from typing import Union, Tuple, Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear
from torch import nn

import abc


class AbstractBaseline(abc.ABC, nn.Module):
    def epoch_log(self, epoch):
        pass


class MLP(nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        activation,
        **kwargs,
    ):
        super(MLP, self).__init__()

        if n_layers == 1:
            modules = [nn.Dropout(p=dropout), nn.Linear(input_dim, output_dim)]
        else:
            modules = [nn.Dropout(p=dropout), nn.Linear(input_dim, hidden_dim)]

            for i in range(n_layers - 2):
                modules.extend(
                    [
                        activation,
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_dim, hidden_dim),
                    ]
                )

            modules.extend(
                [activation, nn.Dropout(p=dropout), nn.Linear(hidden_dim, output_dim)]
            )

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class BasicGNN(torch.nn.Module):
    r"""An abstract class for implementing basic GNN models.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden and output sample.
        num_layers (int): Number of message passing layers.
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (Callable, optional): The non-linear activation function to use.
            (default: :meth:`torch.nn.ReLU(inplace=True)`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"last"`).
            (default: :obj:`"last"`)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float = 0.0,
        act: Optional[Callable] = ReLU(inplace=True),
        norm: Optional[torch.nn.Module] = None,
        jk: str = "last",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = hidden_channels
        if jk == "cat":
            self.out_channels = num_layers * hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = act

        self.convs = ModuleList()

        self.jk = None
        if jk != "last":
            self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)

        self.norms = None
        if norm is not None:
            self.norms = ModuleList([copy.deepcopy(norm) for _ in range(num_layers)])

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()
        if self.jk is not None:
            self.jk.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, *args, **kwargs) -> Tensor:
        xs: List[Tensor] = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, *args, **kwargs)
            if self.norms is not None:
                x = self.norms[i](x)
            if self.act is not None:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.jk is not None:
                xs.append(x)
        return x if self.jk is None else self.jk(xs)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, num_layers={self.num_layers})"
        )


class GCN(BasicGNN):
    r"""The Graph Neural Network from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper, using the
    :class:`~torch_geometric.nn.conv.GCNConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        num_layers (int): Number of GNN layers.
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (Callable, optional): The non-linear activation function to use.
            (default: :meth:`torch.nn.ReLU(inplace=True)`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"last"`).
            (default: :obj:`"last"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GCNConv`.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float = 0.0,
        act: Optional[Callable] = ReLU(inplace=True),
        norm: Optional[torch.nn.Module] = None,
        jk: str = "last",
        **kwargs,
    ):
        super().__init__(
            in_channels, hidden_channels, num_layers, dropout, act, norm, jk
        )

        self.convs.append(GCNConv(in_channels, hidden_channels, **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, **kwargs))


class GraphSAGE(BasicGNN):
    r"""The Graph Neural Network from the `"Inductive Representation Learning
    on Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, using the
    :class:`~torch_geometric.nn.SAGEConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        num_layers (int): Number of GNN layers.
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (Callable, optional): The non-linear activation function to use.
            (default: :meth:`torch.nn.ReLU(inplace=True)`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"last"`).
            (default: :obj:`"last"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.SAGEConv`.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float = 0.0,
        act: Optional[Callable] = ReLU(inplace=True),
        norm: Optional[torch.nn.Module] = None,
        jk: str = "last",
        **kwargs,
    ):
        super().__init__(
            in_channels, hidden_channels, num_layers, dropout, act, norm, jk
        )

        self.convs.append(SAGEConv(in_channels, hidden_channels, **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, **kwargs))


class GIN(BasicGNN):
    r"""The Graph Neural Network from the `"How Powerful are Graph Neural
    Networks?" <https://arxiv.org/abs/1810.00826>`_ paper, using the
    :class:`~torch_geometric.nn.GINConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        num_layers (int): Number of GNN layers.
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (Callable, optional): The non-linear activation function to use.
            (default: :meth:`torch.nn.ReLU(inplace=True)`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"last"`).
            (default: :obj:`"last"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GINConv`.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float = 0.0,
        act: Optional[Callable] = ReLU(inplace=True),
        norm: Optional[torch.nn.Module] = None,
        jk: str = "last",
        **kwargs,
    ):
        super().__init__(
            in_channels, hidden_channels, num_layers, dropout, act, norm, jk
        )

        self.convs.append(GINConv(GIN.MLP(in_channels, hidden_channels), **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(
                GINConv(GIN.MLP(hidden_channels, hidden_channels), **kwargs)
            )

    @staticmethod
    def MLP(in_channels: int, out_channels: int) -> torch.nn.Module:
        return Sequential(
            Linear(in_channels, out_channels),
            BatchNorm1d(out_channels),
            ReLU(inplace=True),
            Linear(out_channels, out_channels),
        )


class GAT(BasicGNN):
    r"""The Graph Neural Network from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper, using the
    :class:`~torch_geometric.nn.GATConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        num_layers (int): Number of GNN layers.
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (Callable, optional): The non-linear activation function to use.
            (default: :meth:`torch.nn.ReLU(inplace=True)`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"last"`).
            (default: :obj:`"last"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GATConv`.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float = 0.0,
        act: Optional[Callable] = ReLU(inplace=True),
        norm: Optional[torch.nn.Module] = None,
        jk: str = "last",
        **kwargs,
    ):
        super().__init__(
            in_channels, hidden_channels, num_layers, dropout, act, norm, jk
        )

        if "concat" in kwargs:
            del kwargs["concat"]

        if "heads" in kwargs:
            assert hidden_channels % kwargs["heads"] == 0
        out_channels = hidden_channels // kwargs.get("heads", 1)

        self.convs.append(GATConv(in_channels, out_channels, dropout=dropout, **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(GATConv(hidden_channels, out_channels, **kwargs))


# From me


class GINGATConv(GATConv):
    def __init__(
        self,
        nn: Callable,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True,
        **kwargs,
    ):

        super(GINGATConv, self).__init__(
            in_channels,
            out_channels,
            heads,
            concat,
            negative_slope,
            dropout,
            add_self_loops,
            bias,
            **kwargs,
        )

        self.nn = nn

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        size: Size = None,
        return_attention_weights=None,
    ):
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        out = super(GINGATConv, self).forward(
            x, edge_index, size, return_attention_weights
        )

        if isinstance(out, tuple):
            out[0] = self.nn(out)
        else:
            out = self.nn(out)

        return out


class GINGAT(BasicGNN):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float = 0.0,
        act: Optional[Callable] = ReLU(inplace=True),
        norm: Optional[torch.nn.Module] = None,
        jk: str = "last",
        **kwargs,
    ):
        super().__init__(
            in_channels, hidden_channels, num_layers, dropout, act, norm, jk
        )

        if "concat" in kwargs:
            del kwargs["concat"]

        if "heads" in kwargs:
            assert hidden_channels % kwargs["heads"] == 0
        out_channels = hidden_channels // kwargs.get("heads", 1)

        self.convs.append(
            GINGATConv(
                GINGAT.MLP(hidden_channels, hidden_channels),
                in_channels,
                out_channels,
                dropout=dropout,
                **kwargs,
            )
        )
        for _ in range(1, num_layers):
            self.convs.append(
                GINGATConv(
                    GINGAT.MLP(hidden_channels, hidden_channels),
                    hidden_channels,
                    out_channels,
                    **kwargs,
                )
            )

    @staticmethod
    def MLP(in_channels: int, out_channels: int) -> torch.nn.Module:
        return Sequential(
            Linear(in_channels, out_channels),
            BatchNorm1d(out_channels),
            ReLU(inplace=True),
            Linear(out_channels, out_channels),
        )


class Symlog(nn.Module):
    def __init__(
        self,
        inplace=False,
        base=torch.tensor(2.71828182845904523536028747135266249775724709369995),
    ):
        super().__init__()
        self.inplace = inplace
        self.logbase = nn.Parameter(torch.log(base), requires_grad=False)  # maybe grad?

    def forward(self, x):
        is_neg = x < -1
        is_pos = x > 1

        if self.inplace:
            x[is_neg] = -(torch.log(-x[is_neg]) / self.logbase) - 1
            x[is_pos] = (torch.log(x[is_pos]) / self.logbase) + 1
            return x
        else:
            x_ = torch.empty_like(x)
            is_mid = abs(x) < 1

            x_[is_neg] = -(torch.log(-x[is_neg]) / self.logbase) - 1
            x_[is_pos] = (torch.log(x[is_pos]) / self.logbase) + 1
            x_[is_mid] = x[is_mid]

            return x_

