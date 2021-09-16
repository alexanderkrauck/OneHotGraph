"""
Implementation of a Sinkhorn Attention Graph
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "21-08-2021"


from typing import Optional, Callable, Union, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from torch_geometric.nn.conv import GATConv
from torch_geometric.utils import softmax
from torch_geometric.typing import OptTensor
from torch_scatter.scatter import scatter_add

from utils.basic_modules import BasicGNN
from torch_geometric.nn.conv.gcn_conv import gcn_norm


# https://github.com/btaba/sinkhorn_knopp
def sinkhorn(
    P: Tensor, threshhold=1e-3, max_iter=100, return_extra=False, do_n_iters=-1
):
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
    if do_n_iters == -1:
        P_eps = P.detach().clone()

    N = P_.shape[1]
    max_thresh = 1 + threshhold
    min_thresh = 1 - threshhold

    # Initialize r and c, the diagonals of D1 and D2
    r = torch.ones((P_.shape[0], N), device=device)

    pdotr = torch.bmm(torch.transpose(P_, 1, 2), r.unsqueeze(-1)).squeeze(-1)

    c = 1 / pdotr
    pdotc = torch.bmm(P_, c.unsqueeze(-1)).squeeze(-1)

    r = 1 / pdotc
    del pdotr, pdotc

    iterations = 0
    stopping_condition = None

    cont = True

    while stopping_condition is None:

        c = 1 / (torch.bmm(torch.transpose(P_, 1, 2), r.unsqueeze(-1)).squeeze(-1))
        r = 1 / (torch.bmm(P_, c.unsqueeze(-1)).squeeze(-1))

        if do_n_iters == -1:
            P_eps = P_ * c.unsqueeze(2)
            P_eps = P_eps * r.unsqueeze(1)

            if (
                not torch.any(torch.sum(P_eps, dim=2) < min_thresh)
                or torch.any(torch.sum(P_eps, dim=2) > max_thresh)
                or torch.any(torch.sum(P_eps, dim=1) < min_thresh)
                or torch.any(torch.sum(P_eps, dim=1) > max_thresh)
            ):
                stopping_condition = "epsilon"
                break

        iterations += 1

        if iterations >= max_iter:
            stopping_condition = "max_iter"
            break

        if iterations == do_n_iters:
            stopping_condition = "done_n_iter"
            break

    P = P * c.unsqueeze(2)
    P = P * r.unsqueeze(1)
    # P = torch.transpose(torch.transpose(P * c, 1, 2) * r, 1, 2)

    if return_extra:
        return P, r, c, stopping_condition
    else:
        return P


class SinkhornGATConv(GATConv):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True,
        norm_mode="sinkhorn",
        do_n_sinkhorn_iters=-1,
        **kwargs
    ):

        keep_keys = ["aggr", "flow", "node_dim"]
        new_kwargs = {}
        for key in keep_keys:
            if key in kwargs:
                new_kwargs[key] = kwargs[key]

        super(SinkhornGATConv, self).__init__(
            in_channels,
            out_channels,
            heads,
            concat,
            negative_slope,
            dropout,
            add_self_loops,
            bias,
            **new_kwargs
        )

        self.norm_mode = norm_mode
        self.do_n_sinkhorn_iters = do_n_sinkhorn_iters

    def message(
        self,
        x_j: Tensor,
        alpha_j: Tensor,
        alpha_i: OptTensor,
        edge_index_i: Tensor,
        edge_index_j: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
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
        # alpha_old = alpha

        if self.norm_mode == "sinkhorn" or self.norm_mode == "gcn":
            alpha = torch.exp(alpha)
        if self.norm_mode == "gcn":
            edge_index = torch.stack((edge_index_j, edge_index_i), dim = 0)
            edge_index, alpha = gcn_norm(edge_index, alpha, size_i, add_self_loops=False)
        if self.norm_mode == "gcn2":
            edge_index = torch.stack((edge_index_j, edge_index_i), dim = 0)
            edge_index, edge_weight = gcn_norm(edge_index, edge_weight=None, num_nodes = size_i, add_self_loops=False)
            alpha = softmax(alpha, edge_index_i, ptr, size_i)
            #TODO: Maybe softmax after the reweighting. But then the impact of gcnnorm is exponential -> also bad. Maybe just a linear normalization (i.e. divide thru scatter add)
            alpha = alpha * edge_weight.unsqueeze(-1)
            renormalizer = (scatter_add(alpha, edge_index_i, dim=0, dim_size=size_i)[edge_index_i])
            alpha = alpha / renormalizer
        if self.norm_mode == "sinkhorn" or self.norm_mode == "both":
            # Sinkhorn Part(This is rather slow) -> TODO: Make it a pytorch-module
            z = torch.zeros((size_i, size_i, alpha.shape[1]), device=alpha.device)
            z[edge_index_j, edge_index_i] = alpha  # maybe switch j and i here?
            z = torch.transpose(z, 0, 2)
            z = z + torch.eye(size_i, device=alpha.device) * 1e-8  # Numerical stabilty

            new_z = sinkhorn(
                z, do_n_iters=self.do_n_sinkhorn_iters
            )  # TODO: by using sparse matrices it might be much faster

            new_z = torch.transpose(z, 0, 2)

            alpha = new_z[edge_index_j, edge_index_i]
            # alpha = alpha

        # Dropout after Sinkhorn!
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.unsqueeze(-1)


class SinkhornGAT(BasicGNN):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float = 0.0,
        act: Optional[Callable] = torch.nn.ReLU(inplace=True),
        norm: Optional[torch.nn.Module] = None,
        jk: str = "last",
        **kwargs
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
            SinkhornGATConv(in_channels, out_channels, dropout=dropout, **kwargs)
        )
        for _ in range(1, num_layers):
            self.convs.append(SinkhornGATConv(hidden_channels, out_channels, **kwargs))

