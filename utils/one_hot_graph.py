
"""
Implementation of the OneHotGraph
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "21-08-2021"

from numpy.core.numeric import ones_like
from torch._C import device
from utils.basic_modules import BasicGNN
from torch_scatter import gather_csr, scatter, segment_csr
from torch_sparse import coalesce



import torch

from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType, OptTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros

from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU

from typing import Optional, Callable, List



class OneHotConv(MessagePassing):

    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(OneHotConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], onehot: Tensor, edge_index: Adj, size: Size = None):

        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        assert x_l.shape[0] == x.shape[0]#mine
        onehot = onehot.unsqueeze(1)
        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out, onehot_out = self.propagate(edge_index, onehot = onehot, x=(x_l, x_r), alpha=(alpha_l, alpha_r), size=size)
        onehot_out.squeeze(1)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        return out, onehot_out

    @staticmethod
    def sparse_dense_mul(s, d):#multiplies along the first (index=0) dim
        i = s._indices()
        v = s._values()
        dv = d[i[0,:]]  # get values from relevant entries of dense matrix
        return torch.sparse.FloatTensor(i, v * dv, s.size())

    def message(self, x_j: Tensor, onehot_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        #onehot_j = OneHotConv.sparse_dense_mul(onehot_j, alpha.squeeze())
        x_j = x_j * alpha.unsqueeze(-1)
        onehot_j = onehot_j * alpha.unsqueeze(-1)

        return x_j, onehot_j #maybe use attention here?

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

    def aggregate(self, inputs: Tensor, index: Tensor, ptr: Optional[Tensor], dim_size: Optional[int]) -> Tensor:
        x_j, onehot_j = inputs 

        aggregated_x_j = scatter(x_j, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        aggregated_onehot_j = scatter(onehot_j, index, dim=self.node_dim, dim_size=dim_size, reduce="add")



        return aggregated_x_j, aggregated_onehot_j

    def update(self, inputs: Tensor, onehot: SparseTensor) -> Tensor:
        aggregated_x_j, aggregated_onehot_j = inputs

        return aggregated_x_j, aggregated_onehot_j + onehot




class OneHotGraph(BasicGNN):

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
                 dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True),
                 norm: Optional[torch.nn.Module] = None, jk: str = 'last',
                 **kwargs):
        super().__init__(in_channels, hidden_channels, num_layers, dropout,
                         act, norm, jk)

        if 'concat' in kwargs:
            del kwargs['concat']

        if 'heads' in kwargs:
            assert hidden_channels % kwargs['heads'] == 0
        out_channels = hidden_channels // kwargs.get('heads', 1)

        self.convs.append(
            OneHotConv(in_channels, out_channels, dropout=dropout, **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(OneHotConv(hidden_channels, out_channels, **kwargs))

    def forward(self, x: Tensor, edge_index: Adj, *args, **kwargs) -> Tensor:
        xs: List[Tensor] = []

        #coords = torch.arange(0, len(x), device = x.device).unsqueeze(0).repeat([2, 1])
        #sparse_identity = torch.sparse_coo_tensor(coords, torch.ones(len(x), device = x.device), (len(x), len(x)), device = x.device)
        #one_hot = torch.sparse_coo_tensor(size=(len(x), len(x)), device = x.device)
        n_nodes = len(x)
        device = x.device
        one_hot = torch.eye(n_nodes, device=device)#torch.zeros((n_nodes, n_nodes), device = device)

        for i in range(self.num_layers):

            #print(x.shape, "\n", x)
            #might not be required -> selfloops

            x, one_hot = self.convs[i](x, one_hot, edge_index, *args, **kwargs)
            if self.norms is not None:
                x = self.norms[i](x)
            if self.act is not None:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.jk is not None:
                xs.append(x)
        return x if self.jk is None else self.jk(xs)