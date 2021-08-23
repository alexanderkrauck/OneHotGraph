
"""
Implementation of the OneHotGraph
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "21-08-2021"

from typing import Union, Tuple, Optional, Callable, List
from torch_geometric.typing import (OptPairTensor, Adj, Size, OptTensor)


import torch
from torch import nn, Tensor
from torch_geometric import nn as gnn
import torch.nn.functional as F

from utils.basic_modules import BasicGNN

import torch_scatter
import torch_sparse
import torch_geometric



class OneHotConv(nn.Module):

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):

        super(OneHotConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)

        self.att_l = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = nn.Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

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
            selected_onehot = onehot.index_select(0, adj[0])
            selected_onehots.append(selected_onehot)

        return selected_onehots

    def forward(
        self, 
        x: Union[torch.Tensor, OptPairTensor], 
        onehots: List[Tensor], 
        edge_index: Adj, 
        batch_sample_indices: Tensor,
        n_sample_nodes: Tensor,
        adjs: List[Tensor], 
        size: Size = None
        ):

        H, C = self.heads, self.out_channels

        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        
        x = self.lin(x).view(-1, H, C)
        alpha_l = (x * self.att_l).sum(dim=-1)
        alpha_r = (x * self.att_r).sum(dim=-1)


        if self.add_self_loops:
            num_nodes = n_sample_nodes.sum()

            edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
            edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=num_nodes)

            for idx in range(len(n_sample_nodes)):
                adj = adjs[idx]
                n_nodes = n_sample_nodes[idx]

                adj, _  = torch_geometric.utils.remove_self_loops(adj)
                adj, _  = torch_geometric.utils.add_self_loops(adj, num_nodes = n_nodes)

                adjs[idx] = adj


        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        x, onehots = self.propagate(
            index = edge_index, 
            onehots = onehots, 
            sample_indices = batch_sample_indices, 
            x = x, 
            alphas = (alpha_l, alpha_r), 
            size = size,
            n_sample_nodes = n_sample_nodes,
            adjs = adjs
        )

        if self.concat:
            x = x.view(-1, self.heads * self.out_channels)
        else:
            x = x.mean(dim=1)

        if self.bias is not None:
            x += self.bias

        return x, onehots

    def propagate(self, x, onehots, index, sample_indices, n_sample_nodes, adjs, alphas, size, **kwargs):
        sending_indices = index[0]
        receiving_indices = index[1]

        sending_alphas = alphas[0]
        receiving_alphas = alphas[1]

        selected_onehots = self.index_select_onehots(onehots, adjs)
        selected_x = x.index_select(0, sending_indices)
        sending_alphas = sending_alphas.index_select(0, sending_indices)
        receiving_alphas = receiving_alphas.index_select(0, sending_indices)#maybe wrong?

        weighted_selected_x, weighted_selected_onethots = self.message(
            selected_x = selected_x, 
            selected_onehots = selected_onehots, 
            sample_indices = sample_indices, 
            n_sample_nodes = n_sample_nodes, 
            selected_alphas = sending_alphas,
            receiving_alphas = receiving_alphas, 
            receiving_index = receiving_indices
        )

        aggregated_selected_x, aggregated_selected_onehots = self.aggregate(
            weighted_selected_x, 
            weighted_selected_onethots, 
            receiving_indices,
            adjs,
            n_sample_nodes
        )

        new_x, new_onehots = self.update(
            aggregated_selected_x,
            aggregated_selected_onehots
        )

        return new_x, new_onehots

    def message(
        self, 
        selected_x: Tensor, 
        selected_onehots: Tensor, 
        sample_indices: Tensor,
        n_sample_nodes: Tensor,
        selected_alphas: Tensor, 
        receiving_alphas: OptTensor,
        receiving_index: Tensor, 
        **kwargs
        ) -> Tensor:

        alpha = selected_alphas if receiving_alphas is None else selected_alphas + receiving_alphas
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = torch_geometric.utils.softmax(alpha, receiving_index)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        weighted_selected_x = selected_x * alpha.unsqueeze(-1)

        cumsumed_n_sample_nodes = torch.tensor([len(l) for l in selected_onehots], device = selected_x.device).cumsum(0)
        weighted_selected_onehots = []
        for idx in range(len(selected_onehots)):
            if idx == 0:
                idx0 = 0
            else:
                idx0 = cumsumed_n_sample_nodes[idx - 1]
            idx1 = cumsumed_n_sample_nodes[idx]
            weighted_selected_onehots.append(selected_onehots[idx] * alpha[idx0: idx1])

        return weighted_selected_x, weighted_selected_onehots

    def aggregate(
        self,
        weighted_selected_x, 
        weighted_selected_onethots, 
        receiving_indices: Tensor, 
        adjs,
        n_sample_nodes: Tensor
        ) -> Tensor:

        aggregated_selected_x = torch_scatter.scatter(weighted_selected_x, receiving_indices, dim=0, dim_size=n_sample_nodes.sum(), reduce="sum")
        
        aggregated_selected_onehots = []
        for ws_onehot, adj, n_nodes in zip(weighted_selected_onethots, adjs, n_sample_nodes):
            aggregated_selected_onehots.append(torch_scatter.scatter(ws_onehot, adj[1], dim=0, dim_size=n_nodes, reduce="add"))

        return aggregated_selected_x, aggregated_selected_onehots

    def update(self, aggregated_selected_x: Tensor, aggregated_selected_onehots: List[Tensor]) -> Tensor:

        return aggregated_selected_x, aggregated_selected_onehots





class OneHotGraph(BasicGNN):

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
                 dropout: float = 0.0,
                 act: Optional[Callable] = nn.ReLU(inplace=True),
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

    def forward(self, x: Tensor, edge_index: Adj, batch_sample_indices, n_sample_nodes, adjs, *args, **kwargs) -> Tensor:

        xs: List[Tensor] = []
        device = x.device

        one_hots = []
        for n in n_sample_nodes:
            one_hots.append(torch.eye(n, device=device))

        for i in range(self.num_layers):
            x, one_hot = self.convs[i](x, one_hots, edge_index, batch_sample_indices, n_sample_nodes, adjs, *args, **kwargs)
            if self.norms is not None:
                x = self.norms[i](x)
            if self.act is not None:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.jk is not None:
                xs.append(x)
        return x if self.jk is None else self.jk(xs)