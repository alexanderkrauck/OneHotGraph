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
from torch_geometric.nn.conv import MessagePassing
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


#https://github.com/btaba/sinkhorn_knopp

import warnings

import numpy as np


class SinkhornKnopp:
    """
    Sinkhorn Knopp Algorithm

    Takes a non-negative square matrix P, where P =/= 0
    and iterates through Sinkhorn Knopp's algorithm
    to convert P to a doubly stochastic matrix.
    Guaranteed convergence if P has total support.

    For reference see original paper:
        http://msp.org/pjm/1967/21-2/pjm-v21-n2-p14-s.pdf

    Parameters
    ----------
    max_iter : int, default=1000
        The maximum number of iterations.

    epsilon : float, default=1e-3
        Metric used to compute the stopping condition,
        which occurs if all the row and column sums are
        within epsilon of 1. This should be a very small value.
        Epsilon must be between 0 and 1.

    Attributes
    ----------
    _max_iter : int, default=1000
        User defined parameter. See above.

    _epsilon : float, default=1e-3
        User defined paramter. See above.

    _stopping_condition: string
        Either "max_iter", "epsilon", or None, which is a
        description of why the algorithm stopped iterating.

    _iterations : int
        The number of iterations elapsed during the algorithm's
        run-time.

    _D1 : 2d-array
        Diagonal matrix obtained after a stopping condition was met
        so that _D1.dot(P).dot(_D2) is close to doubly stochastic.

    _D2 : 2d-array
        Diagonal matrix obtained after a stopping condition was met
        so that _D1.dot(P).dot(_D2) is close to doubly stochastic.

    Example
    -------

    .. code-block:: python
        >>> import numpy as np
        >>> from sinkhorn_knopp import sinkhorn_knopp as skp
        >>> sk = skp.SinkhornKnopp()
        >>> P = [[.011, .15], [1.71, .1]]
        >>> P_ds = sk.fit(P)
        >>> P_ds
        array([[ 0.06102561,  0.93897439],
           [ 0.93809928,  0.06190072]])
        >>> np.sum(P_ds, axis=0)
        array([ 0.99912489,  1.00087511])
        >>> np.sum(P_ds, axis=1)
        array([ 1.,  1.])

    """

    def __init__(self, max_iter=1000, epsilon=1e-3):
        assert isinstance(max_iter, int) or isinstance(max_iter, float),\
            "max_iter is not of type int or float: %r" % max_iter
        assert max_iter > 0,\
            "max_iter must be greater than 0: %r" % max_iter
        self._max_iter = int(max_iter)

        assert isinstance(epsilon, int) or isinstance(epsilon, float),\
            "epsilon is not of type float or int: %r" % epsilon
        assert epsilon > 0 and epsilon < 1,\
            "epsilon must be between 0 and 1 exclusive: %r" % epsilon
        self._epsilon = epsilon

        self._stopping_condition = None
        self._iterations = 0
        self._D1 = np.ones(1)
        self._D2 = np.ones(1)

    def fit(self, P):
        """Fit the diagonal matrices in Sinkhorn Knopp's algorithm

        Parameters
        ----------
        P : 2d array-like
        Must be a square non-negative 2d array-like object, that
        is convertible to a numpy array. The matrix must not be
        equal to 0 and it must have total support for the algorithm
        to converge.

        Returns
        -------
        A double stochastic matrix.

        """
        P = np.asarray(P)
        assert np.all(P >= 0)
        assert P.ndim == 2
        assert P.shape[0] == P.shape[1]

        N = P.shape[0]
        max_thresh = 1 + self._epsilon
        min_thresh = 1 - self._epsilon

        # Initialize r and c, the diagonals of D1 and D2
        # and warn if the matrix does not have support.
        r = np.ones((N, 1))
        pdotr = P.T.dot(r)
        total_support_warning_str = (
            "Matrix P must have total support. "
            "See documentation"
        )
        if not np.all(pdotr != 0):
            warnings.warn(total_support_warning_str, UserWarning)

        c = 1 / pdotr
        pdotc = P.dot(c)
        if not np.all(pdotc != 0):
            warnings.warn(total_support_warning_str, UserWarning)

        r = 1 / pdotc
        del pdotr, pdotc

        P_eps = np.copy(P)
        while np.any(np.sum(P_eps, axis=1) < min_thresh) \
                or np.any(np.sum(P_eps, axis=1) > max_thresh) \
                or np.any(np.sum(P_eps, axis=0) < min_thresh) \
                or np.any(np.sum(P_eps, axis=0) > max_thresh):

            c = 1 / P.T.dot(r)
            r = 1 / P.dot(c)

            self._D1 = np.diag(np.squeeze(r))
            self._D2 = np.diag(np.squeeze(c))
            P_eps = self._D1.dot(P).dot(self._D2)

            self._iterations += 1

            if self._iterations >= self._max_iter:
                self._stopping_condition = "max_iter"
                break

        if not self._stopping_condition:
            self._stopping_condition = "epsilon"

        self._D1 = np.diag(np.squeeze(r))
        self._D2 = np.diag(np.squeeze(c))
        P_eps = self._D1.dot(P).dot(self._D2)

        return P_eps

class SinkhornGATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(SinkhornGATConv, self).__init__(node_dim=0, **kwargs)

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

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None):
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
        H, C = self.heads, self.out_channels
        #pdb.set_trace()

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

        #Self loops might be that nodes are also connected to themselfes
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

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        #print("x_l:\n",x_l)
        #print("x_r:\n",x_r)
        #print("alpha_l:\n",alpha_l)
        #print("alpha_r:\n", alpha_r)
        #print("size:\n",size)
        #print("edge_index:\n", edge_index)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

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
        #index: contains the index where each x_j and alpha goes to (where the message is sent to)
        #We now need also where the message comes from to make the matrix double stochastic! (need to add it)
        #This is contained in the edge_index matrix
        
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training) 
        #print("alpha:\n", alpha)

        #print("alpha_shape:\n",alpha.shape)
        #print("x_j:\n",x_j)
        #print("x_j shape", x_j.shape)
        #print("edge_index_i:\n", edge_index_i)
        #print("ptr\n",ptr)
        #print("size_i\n",size_i)
        #print("edge_index_j:\n",edge_index_j)
        z = torch.zeros((size_i, size_i))
        z[edge_index_j, edge_index_i] = alpha.squeeze()#maybe switch j and i here?
        sk = SinkhornKnopp()
        _ = sk.fit(z.detach().cpu().numpy())
        D1 = torch.tensor(sk._D1, dtype=torch.float32)
        D2 = torch.tensor(sk._D2, dtype=torch.float32)
        new_z = D1 @ z @ D2
        alpha = new_z[edge_index_j, edge_index_i]
        alpha = alpha.unsqueeze(-1)
        #print("alpha_new:\n",alpha)
        #TODO: reparametize here
        #print("z new:\n",new_z)

        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

from typing import Optional, Callable, List
from torch_geometric.typing import Adj

import copy

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU

from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge

from utils.basic_modules import BasicGNN

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