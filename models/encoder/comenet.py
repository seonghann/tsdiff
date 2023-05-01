from torch_cluster import radius_graph
from torch_geometric.nn import GraphConv, GraphNorm
#from torch_geometric.nn.acts import swish
from utils import swish
from torch_geometric.nn import inits
from torch_geometric.nn.conv import MessagePassing

# from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size

from typing import Tuple, Union
from ..geometry import angle_emb, torsion_emb
from torch_scatter import scatter, scatter_min

from torch.nn import Embedding
from torch_sparse import SparseTensor, matmul

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

import math
from math import sqrt

try:
    import sympy as sym
except ImportError:
    sym = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomLinear(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=True,
        weight_initializer="glorot",
        bias_initializer="zeros",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        assert in_channels > 0
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.in_channels > 0:
            if self.weight_initializer == "glorot":
                inits.glorot(self.weight)
            elif self.weight_initializer == "glorot_orthogonal":
                inits.glorot_orthogonal(self.weight, scale=2.0)
            elif self.weight_initializer == "uniform":
                bound = 1.0 / math.sqrt(self.weight.size(-1))
                torch.nn.init.uniform_(self.weight.data, -bound, bound)
            elif self.weight_initializer == "kaiming_uniform":
                inits.kaiming_uniform(self.weight, fan=self.in_channels, a=math.sqrt(5))
            elif self.weight_initializer == "zeros":
                inits.zeros(self.weight)
            elif self.weight_initializer is None:
                inits.kaiming_uniform(self.weight, fan=self.in_channels, a=math.sqrt(5))
            else:
                raise RuntimeError(
                    f"Linear layer weight initializer "
                    f"'{self.weight_initializer}' is not supported"
                )

        if self.in_channels > 0 and self.bias is not None:
            if self.bias_initializer == "zeros":
                inits.zeros(self.bias)
            elif self.bias_initializer is None:
                inits.uniform(self.in_channels, self.bias)
            else:
                raise RuntimeError(
                    f"Linear layer bias initializer "
                    f"'{self.bias_initializer}' is not supported"
                )

    def forward(self, x):
        """"""
        return F.linear(x, self.weight, self.bias)


class TwoLayerLinear(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        middle_channels,
        out_channels,
        dropout=0.1,
        bias=True,
        act=True,
    ):
        super(TwoLayerLinear, self).__init__()
        # self.lin1 = Linear(in_channels, middle_channels, bias=bias)
        # self.lin2 = Linear(middle_channels, out_channels, bias=bias)
        self.lin1 = nn.Sequential(
            CustomLinear(in_channels, middle_channels, bias=bias), nn.Dropout(dropout)
        )
        self.lin2 = nn.Sequential(
            CustomLinear(middle_channels, out_channels, bias=bias), nn.Dropout(dropout)
        )
        self.act = act

    def reset_parameters(self):
        self.lin1[0].reset_parameters()
        self.lin2[0].reset_parameters()

    def forward(self, x):
        x = self.lin1(x)

        if self.act:
            x = swish(x)
        x = self.lin2(x)
        if self.act:
            x = swish(x)
        return x


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super(EmbeddingBlock, self).__init__()
        self.act = act
        self.emb = Embedding(100, hidden_channels)
        # self.emb = CustomLinear(inp, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))

    def forward(self, x):
        x = self.act(self.emb(x))
        return x


class EdgeGraphConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: str = "add",
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_rel = CustomLinear(in_channels[0], out_channels, bias=bias)
        self.lin_root = CustomLinear(in_channels[1], out_channels, bias=False)

        self.edge_lin_1 = CustomLinear(self.in_channels * 2, out_channels)
        self.act = swish
        self.edge_lin_2 = CustomLinear(out_channels, out_channels)

        self.edge_attn_1 = CustomLinear(self.in_channels, 1)
        self.sigmoid = nn.Sigmoid()

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()
        self.edge_lin_1.reset_parameters()
        self.edge_lin_2.reset_parameters()
        self.edge_attn_1.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_bond_attr: OptTensor = None,
        edge_geom_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(
            edge_index,
            x=x,
            edge_bond_attr=edge_bond_attr,
            edge_geom_attr=edge_geom_attr,
            size=size,
        )
        out = self.lin_rel(out)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_root(x_r)

        return out

    def message(self, x_j, edge_bond_attr, edge_geom_attr) -> Tensor:
        edge_weight = torch.cat([edge_bond_attr, edge_geom_attr], dim=1)
        edge_weight = self.edge_lin_1(edge_weight)
        edge_weight = self.edge_lin_2(self.act(edge_weight))

        x_j = edge_weight * x_j
        attn = self.sigmoid(self.edge_attn_1(x_j))
        x_j = x_j * attn
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: OptPairTensor) -> Tensor:
        return matmul(adj_t, x[0], reduce=self.aggr)


class SimpleInteractionBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        num_layers,
        output_channels,
        dropout=0.1,
        act=swish,
        inits="glorot",
    ):
        super(SimpleInteractionBlock, self).__init__()
        self.act = act
        self.conv1 = EdgeGraphConv(hidden_channels, hidden_channels)
        self.conv2 = EdgeGraphConv(hidden_channels, hidden_channels)
        self.lin1 = CustomLinear(hidden_channels, hidden_channels)
        self.lin2 = CustomLinear(hidden_channels, hidden_channels)
        self.lin_cat = CustomLinear(2 * hidden_channels, hidden_channels)
        self.norm = GraphNorm(hidden_channels)

        # Transformations of Bessel and spherical basis representations.
        # Dense transformations of input messages.
        self.lin = CustomLinear(hidden_channels, hidden_channels)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(CustomLinear(hidden_channels, hidden_channels))
            self.lins.append(nn.Dropout(dropout))
        self.final = CustomLinear(
            hidden_channels, output_channels, weight_initializer=inits
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        self.norm.reset_parameters()

        self.lin.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        self.lin_cat.reset_parameters()
        i = 0
        for lin in self.lins:
            i += 1
            if i % 2 == 1:
                lin.reset_parameters()

        self.final.reset_parameters()

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        edge_geom_attr1,
        edge_geom_attr2,
        batch,
    ):
        x = self.act(self.lin(x))

        h1 = self.conv1(x, edge_index, edge_attr, edge_geom_attr1)
        h1 = self.lin1(h1)
        h1 = self.act(h1)

        h2 = self.conv2(x, edge_index, edge_attr, edge_geom_attr2)
        h2 = self.lin2(h2)
        h2 = self.act(h2)

        h = self.lin_cat(torch.cat([h1, h2], 1))
        h = h + x
        for lin in self.lins:
            h = self.act(lin(h)) + h
        h = self.norm(h, batch)
        h = self.final(h)
        return h


def get_pi_npi_features(pos, i, j, batch):
    """
    i : non-pi atom index
    j : pi atom index

    1. atom_j and atom_i has no bond.
    2. topological distance between atom_i and atom_j is ...
    3. atom_j has sp2 hybridzation or has pi-bond (conjugated/double/triple/aromatic bond)

    to consider for long range pi interaction,
    feature should contains pi-bond orbital geometry.

    atom_j is set to origin.
    z-axis is defined as normal vector of the plane constructed by
    two vectors, which are heading to nearlest neighbor atoms atom_j0 atom_j1.
    Then consider angle between z-axis and vector atom_j_to_atom_i

    Positive direction of z_axis is set toward atom_i.
    """

    # set z-axis
    # j need to be sorted
    # nearlest neighbor
    query_index, nei_index = radius_graph(pos, 3, batch=batch)
    nei_index = nei_index[torch.isin(query_index, j)]
    query_j_index = query_index[torch.isin(query_index, j)]

    query_dist = (pos[query_j_index] - pos[nei_index]).norm(dim=-1)
    _, argmin0 = scatter_min(query_dist, query_j_index)
    argmin0[argmin0 < len(query_j_index)]
    nei0 = nei_index[argmin0]

    add = torch.zeros_like(query_dist).to(query_dist.device)
    add[nei0] = torch.inf
    query_dist1 = query_dist + add

    _, argmin1 = scatter_min(query_dist1, query_j_index)
    argmin1[argmin1 < len(query_j_index)]
    nei1 = nei_index[argmin1]

    query_j = torch.unique(j, sorted=True)

    v1 = pos[nei0] - pos[query_j]
    v2 = pos[nei1] - pos[query_j]
    z_axis_src = torch.cross(v1, v2, dim=-1)

    z_axis_idx = [(j.unsqueeze(0) == query_j.unsqueeze(1)).nonzero()][:, 0]
    z_axis = z_axis_src[z_axis_idx]

    vecs = pos[i] - pos[j]
    dist = vecs.norm(dim=-1)
    z_axis = torch.sign((z_axis * vecs).sum(-1)) * z_axis

    a = (vecs * z_axis).sum(dim=-1)
    b = torch.cross(vecs, z_axis).norm(dim=-1)
    theta = torch.atan2(b, a)
    return dist, theta


def get_pi_pi_features(pos, i, j, batch):
    dist1, theta1 = get_pi_npi_features(pos, i, j, batch)
    dist2, theta2 = get_pi_npi_features(pos, j, i, batch)
    return dist1, theta1, dist2, theta2


def get_features(dist, vecs, i, j, num_nodes, cutoff):
    # Calculate distances.
    _, argmin0 = scatter_min(dist, i, dim_size=num_nodes)
    argmin0[argmin0 >= len(i)] = 0
    n0 = j[argmin0]
    add = torch.zeros_like(dist).to(dist.device)
    add[argmin0] = cutoff
    dist1 = dist + add

    _, argmin1 = scatter_min(dist1, i, dim_size=num_nodes)
    argmin1[argmin1 >= len(i)] = 0
    n1 = j[argmin1]
    # --------------------------------------------------------

    _, argmin0_j = scatter_min(dist, j, dim_size=num_nodes)
    argmin0_j[argmin0_j >= len(j)] = 0
    n0_j = i[argmin0_j]

    add_j = torch.zeros_like(dist).to(dist.device)
    add_j[argmin0_j] = cutoff
    dist1_j = dist + add_j

    # i[argmin] = range(0, num_nodes)
    _, argmin1_j = scatter_min(dist1_j, j, dim_size=num_nodes)
    argmin1_j[argmin1_j >= len(j)] = 0
    n1_j = i[argmin1_j]

    # ----------------------------------------------------------

    # n0, n1 for i
    n0 = n0[i]
    n1 = n1[i]

    # n0, n1 for j
    n0_j = n0_j[j]
    n1_j = n1_j[j]

    # tau: (iref, i, j, jref)
    # when compute tau, do not use n0, n0_j as ref for i and j,
    # because if n0 = j, or n0_j = i, the computed tau is zero
    # so if n0 = j, we choose iref = n1
    # if n0_j = i, we choose jref = n1_j
    mask_iref = n0 == j
    iref = torch.clone(n0)
    iref[mask_iref] = n1[mask_iref]
    idx_iref = argmin0[i]
    idx_iref[mask_iref] = argmin1[i][mask_iref]

    mask_jref = n0_j == i
    jref = torch.clone(n0_j)
    jref[mask_jref] = n1_j[mask_jref]
    idx_jref = argmin0_j[j]
    idx_jref[mask_jref] = argmin1_j[j][mask_jref]

    pos_ji, pos_in0, pos_in1, pos_iref, pos_jref_j = (
        vecs,
        vecs[argmin0][i],
        vecs[argmin1][i],
        vecs[idx_iref],
        vecs[idx_jref],
    )

    # Calculate angles.
    a = ((-pos_ji) * pos_in0).sum(dim=-1)
    b = torch.cross(-pos_ji, pos_in0).norm(dim=-1)
    theta = torch.atan2(b, a)
    theta[theta < 0] = theta[theta < 0] + math.pi

    # Calculate torsions.
    dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
    plane1 = torch.cross(-pos_ji, pos_in0)
    plane2 = torch.cross(-pos_ji, pos_in1)
    a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
    b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
    phi = torch.atan2(b, a)
    phi[phi < 0] = phi[phi < 0] + math.pi

    # Calculate right torsions.
    plane1 = torch.cross(pos_ji, pos_jref_j)
    plane2 = torch.cross(pos_ji, pos_iref)
    a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
    b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
    tau = torch.atan2(b, a)
    tau[tau < 0] = tau[tau < 0] + math.pi
    return theta, phi, tau


class ComENetEncoder(nn.Module):
    r"""
     The ComENet from the `"ComENet: Towards Complete and Efficient Message Passing for 3D Molecular Graphs" <https://arxiv.org/abs/2206.08515>`_ paper.

    Args:
        cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`8.0`)
        num_layers (int, optional): Number of building blocks. (default: :obj:`4`)
        hidden_channels (int, optional): Hidden embedding size. (default: :obj:`256`)
        out_channels (int, optional): Size of each output sample. (default: :obj:`1`)
        num_radial (int, optional): Number of radial basis functions. (default: :obj:`3`)
        num_spherical (int, optional): Number of spherical harmonics. (default: :obj:`2`)
        num_output_layers (int, optional): Number of linear layers for the output blocks. (default: :obj:`3`)
    """

    def __init__(
        self,
        cutoff=8.0,
        num_layers=4,
        hidden_channels=256,
        out_channels=1,
        num_radial=3,
        num_spherical=2,
        num_output_layers=3,
        dropout=0.1,
        act=None,
        **kwargs,
    ):
        super(ComENetEncoder, self).__init__()
        self.out_channels = out_channels
        self.cutoff = cutoff
        self.num_layers = num_layers

        if sym is None:
            raise ImportError("Package `sympy` could not be found.")

        self.act = act

        self.feature1 = torsion_emb(
            num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff
        )
        self.feature2 = angle_emb(
            num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff
        )

        self.emb = EmbeddingBlock(hidden_channels, act)
        self.edge_emb = Embedding(100, embedding_dim=hidden_channels)
        self.edge_cat = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels *2, hidden_channels),
                act,
                torch.nn.Linear(hidden_channels, hidden_channels)
                )

        if "lin_feature" in kwargs and kwargs["lin_feature"] is not None:
            self.lin_feature1 = kwargs["lin_feature"][0]
            self.lin_feature2 = kwargs["lin_feature"][1]

        else:
            self.lin_feature1 = TwoLayerLinear(
                num_radial * num_spherical**2,
                hidden_channels,
                hidden_channels,
                dropout,
            )
            self.lin_feature2 = TwoLayerLinear(
                num_radial * num_spherical, hidden_channels, hidden_channels, dropout
            )

        self.interaction_blocks = torch.nn.ModuleList(
            [
                SimpleInteractionBlock(
                    hidden_channels,
                    num_output_layers,
                    hidden_channels,
                    dropout,
                    act,
                )
                for _ in range(num_layers)
            ]
        )

        self.lins = torch.nn.ModuleList()
        for _ in range(num_output_layers):
            self.lins.append(CustomLinear(hidden_channels, hidden_channels))
        self.lin_out = CustomLinear(
            hidden_channels, out_channels, weight_initializer="zeros"
        )
        self.reset_parameters()
    
    @classmethod
    def from_config(cls, config):
        if not config.edge_emb:
            raise Exception(
                    "ComENetEncoder Must have edge-embedding internally\n"
                    "Check config.global(local)_encoder.edge_emb"
                    )

        encoder = cls(
                cutoff=config.cutoff,
                num_layers=config.num_convs,
                hidden_channels=config.hidden_dim,
                num_radial=config.num_radial,
                num_spherical=config.num_spherical,
                dropout=config.dropout,
                )
        return encoder

    def reset_parameters(self):
        self.emb.reset_parameters()
        self.lin_feature1.reset_parameters()
        self.lin_feature2.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        self.lin_out.reset_parameters()

    def get_geom_feat(
        self,
        pos,
        edge_index,
    ):
        j, i = edge_index
        vecs = pos[j] - pos[i]
        dist = vecs.norm(dim=-1)
        num_nodes = pos.size(0)

        theta, phi, tau = get_features(dist, vecs, i, j, num_nodes, self.cutoff)
        return theta, phi, tau

    def forward(self, z, pos, batch, edge_index, edge_type, **kwargs):
        num_nodes = z.size(0)

        # Embedding block.
        # x = self.emb(z)
        x = z

        
        j, i = edge_index
        vecs = pos[j] - pos[i]
        dist = vecs.norm(dim=-1)
        theta, phi, tau = get_features(dist, vecs, i, j, num_nodes, self.cutoff)
        
        edge_type_r, edge_type_p = edge_type
        edge_emb_r = self.edge_emb(edge_type_r)
        edge_emb_p = self.edge_emb(edge_type_p)
        edge_geom_attr1 = self.lin_feature1(self.feature1(dist, theta, phi))
        edge_geom_attr1 = self.edge_cat(
                torch.cat(
                    [edge_geom_attr1 * edge_emb_r, edge_geom_attr1 * edge_emb_p], 
                    dim=-1)
                )

        edge_geom_attr2 = self.lin_feature2(self.feature2(dist, tau))
        edge_geom_attr2 = self.edge_cat(
                torch.cat(
                    [edge_geom_attr2 * edge_emb_r, edge_geom_attr2 * edge_emb_p], 
                    dim=-1)
                )

        # Interaction blocks.
        residual = x
        for interaction_block in self.interaction_blocks:
            _x = interaction_block(
                x,
                edge_index,
                edge_geom_attr1,
                edge_geom_attr2,
                batch,
            )
            x = _x + residual
            residual = residual + _x

        for lin in self.lins:
            x = self.act(lin(x))
        x = self.lin_out(x)
        edges = [edge_geom_attr1, edge_geom_attr2]
        return x, edges
