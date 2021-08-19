
"""
Implementation of a Sinkhorn Attention Graph
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "21-08-2021"

from torch import nn
from utils.basic_modules import GIN, MLP, GAT
from torch_geometric import nn as gnn

#Copied -> TODO: rewrite GAT-Conv to Sinkhorn-Conv 
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing, GATConv
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros




from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros

from typing import Optional, Callable, List
from torch_geometric.typing import Adj

import copy

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU

from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge

from utils.basic_modules import BasicGNN


#https://github.com/btaba/sinkhorn_knopp


import numpy as np



def sinkhorn(P: Tensor, threshhold = 1e-3, max_iter = 100, return_extra = False):
    """Fit the diagonal matrices in Sinkhorn Knopp's algorithm

    Parameters
    ----------
    P : 2d Tensor

    Returns
    -------
    A double stochastic matrix.

    """

    device = P.device
    P_ = P.detach().clone()
    P_eps = P.detach().clone()

    N = P_.shape[0]
    max_thresh = 1 + threshhold
    min_thresh = 1 - threshhold

    # Initialize r and c, the diagonals of D1 and D2
    r = torch.ones(N, device = device)
    pdotr = P_.T @ r

    c = 1 / pdotr
    pdotc = P_ @ c


    r = 1 / pdotc
    del pdotr, pdotc

    iterations = 0
    stopping_condition = None
    while torch.any(torch.sum(P_eps, dim=1) < min_thresh) \
            or torch.any(torch.sum(P_eps, dim=1) > max_thresh) \
            or torch.any(torch.sum(P_eps, dim=0) < min_thresh) \
            or torch.any(torch.sum(P_eps, dim=0) > max_thresh):

        c = 1 / (P_.T @ r)
        r = 1 / (P_ @ c)

        P_eps = ((P_ * c).T * r).T

        iterations += 1

        if iterations >= max_iter:
            stopping_condition = "max_iter"
            break

    if not stopping_condition:
        stopping_condition = "epsilon"


    P = ((P * c).T * r).T

    if return_extra:
        return P, r, c, stopping_condition
    else:
        return P


class SinkhornGATConv(GATConv):

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True, norm_mode = "sinkhorn", **kwargs):

        super(SinkhornGATConv, self).__init__(in_channels, out_channels, heads, concat, negative_slope, dropout, add_self_loops, bias, **kwargs)

        self.norm_mode = norm_mode


    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                edge_index_i: Tensor, edge_index_j: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        """Attention construction

        For the Sinkhorn-Knoff implementation the reparametization trick is used to keep differentiablity

        edge_index_i: Tensor
            The receiving edges of the messages
        edge_index_j: Tensor
            The sourcing edges of the messages
        """
        
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)

        if self.norm_mode == "softmax" or "both":
            alpha = softmax(alpha, edge_index_i, ptr, size_i)
        #alpha_old = alpha

        if self.norm_mode == "sinkhorn":
            alpha = torch.exp(alpha)

        if self.norm_mode == "sinkhorn" or "both":
            #Sinkhorn Part(This is rather slow) -> TODO: Make it a pytorch-module
            z = torch.zeros((size_i, size_i), device = alpha.device)
            z[edge_index_j, edge_index_i] = alpha.squeeze()#maybe switch j and i here?
            z = z + torch.eye(z.shape[0], device = alpha.device) * 1e-8 #Numerical stabilty

            new_z = sinkhorn(z)#TODO: by using sparse matrices it might be much faster


            alpha = new_z[edge_index_j, edge_index_i]
            alpha = alpha.unsqueeze(-1)


        #Dropout after Sinkhorn!
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training) 

        return x_j * alpha.unsqueeze(-1)


class SinkhornGAT(BasicGNN):
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
            SinkhornGATConv(in_channels, out_channels, dropout=dropout, **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(SinkhornGATConv(hidden_channels, out_channels, **kwargs))
    