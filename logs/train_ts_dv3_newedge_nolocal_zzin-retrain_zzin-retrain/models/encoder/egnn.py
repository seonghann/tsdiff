import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, Size, Tensor
from torch_scatter import scatter_mean
from utils import activation_loader


class EGNNMixed2DEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        num_convs=5,
        dropout=0.1,
        initialize_weights=False,
        aggr="mean",
        scatter_fun=scatter_mean,
        pretrain=False,
    ):
        """Main Equivariant Graph Neural Network class.

        Parameters
        ----------
        embedding_dim : int, optional
            Embedding dimension, by default 128
        n_kernels : int, optional
            Number of message-passing rounds, by default 5
        n_mlp : int, optional
            Number of node-level and global-level MLPs, by default 3
        mlp_dim : int, optional
            Hidden size of the node-level and global-level MLPs, by default 256
        n_outputs : int, optional
            Number of endpoints to predict, by default 1
        m_dim : int, optional
            Node-level hidden size, by default 32
        initialize_weights : bool, optional
            Whether to use Xavier init. for learnable weights, by default True
        aggr : str, optional
            Aggregation strategy for global tasks, by default "mean"
        scatter_fun : function, optional
            Which torch.scatter function to use in order to aggregate node-level features
        """
        super(EGNNMixed2DEncoder, self).__init__()

        self.pos_dim = 3
        self.hidden_dim = hidden_dim
        self.num_convs = num_convs
        self.n_outputs = hidden_dim
        self.initialize_weights = initialize_weights
        self.aggr = aggr
        self.scatter_fun = scatter_fun
        self.dropout = dropout

        # Kernel
        self.egnn_kernels = nn.ModuleList()
        for _ in range(self.num_convs):
            self.egnn_kernels.append(
                EGNN_sparse(
                    hidden_dim=self.hidden_dim,
                    pos_dim=self.pos_dim,
                    aggr=self.aggr,
                    dropout=self.dropout,
                )
            )

        self.lincat_kernel = nn.Sequential(
            torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            activation_loader("swish"),
            torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        # self.lincat_kernels = nn.ModuleList()
        # for _ in range(self.num_convs):
        #    __ = nn.Sequential(
        #            torch.nn.Linear(self.hidden_dim*2,self.hidden_dim*2),
        #            activation_loader('swish'),
        #            torch.nn.Linear(self.hidden_dim*2,self.hidden_dim)
        #            )
        #    self.lincat_kernels.append(__)

        self.gin_kernels = nn.ModuleList()
        for _ in range(self.num_convs):
            edge_cat = nn.Sequential(
                torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
                activation_loader("swish"),
                torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            )
            fin_layer = nn.Sequential(
                torch.nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                activation_loader("swish"),
                torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            )
            self.gin_kernels.append(GINEConv(fin_layer, edge_cat, activation="swish"))

    def forward(
        self, node, edge_index_local, edge_attr_r, edge_attr_p, edge_index_global, pos
    ):
        h = node
        for egnn_kernel, gin_kernel in zip(self.egnn_kernels, self.gin_kernels):
            pos, h1 = egnn_kernel(pos=pos, x=h, edge_index=edge_index_global)
            h2 = gin_kernel(
                x=h,
                edge_index=edge_index_local,
                edge_attr_r=edge_attr_r,
                edge_attr_p=edge_attr_p,
            )
            dh = self.lincat_kernel(torch.cat([h1, h2], dim=-1))
            h = h + dh

        return h


class GINEConv(MessagePassing):
    def __init__(
        self,
        nn,
        edge_cat,
        eps: float = 0.0,
        train_eps: bool = False,
        activation="swish",
        **kwargs,
    ):
        super(GINEConv, self).__init__(aggr="add", **kwargs)
        self.nn = nn
        self.initial_eps = eps
        self.edge_cat = edge_cat
        if isinstance(activation, str):
            self.activation = activation_loader(activation)
        else:
            self.activation = None

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))

    def forward(self, x, edge_index, edge_attr_r, edge_attr_p, size=None):
        if isinstance(x, Tensor):
            x = (x, x)

        assert edge_attr_r is not None and edge_attr_p is not None

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(
            edge_index, x=x, edge_attr_r=edge_attr_r, edge_attr_p=edge_attr_p, size=size
        )

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr_r, edge_attr_p: Tensor) -> Tensor:
        edge_attr = self.edge_cat(torch.cat([edge_attr_r, edge_attr_p], dim=-1))
        if self.activation:
            return self.activation(x_j * edge_attr)
        else:
            return x_j * edge_attr

    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.nn)


def weights_init(m):
    """Xavier uniform weight initialization

    Parameters
    ----------
    m : [torch.nn.modules.linear.Linear]
        A list of learnable linear PyTorch modules.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class EGNN_sparse(MessagePassing):
    def __init__(
        self,
        hidden_dim,
        pos_dim=3,
        dropout=0.1,
        aggr="mean",
        **kwargs,
    ):
        """Base torch geometric EGNN message-passing layer

        Parameters
        ----------
        feats_dim : int
            Dimension of the node-level features
        pos_dim : int, optional
            Dimensions of the positional features (e.g. cartesian coordinates), by default 3
        edge_attr_dim : int, optional
            Dimension of the edge-level features, by default 0
        m_dim : int, optional
            Hidden node/edge layer size, by default 32
        dropout : float, optional
            Whether to use dropout, by default 0.1
        aggr : str, optional
            Node update function, by default "mean"
        """
        valid_aggrs = {
            "add",
            "sum",
            "max",
            "mean",
        }
        assert aggr in valid_aggrs, f"pool method must be one of {valid_aggrs}"

        kwargs.setdefault("aggr", aggr)
        super(EGNN_sparse, self).__init__(**kwargs)

        # Model parameters
        self.hidden_dim = hidden_dim
        self.pos_dim = pos_dim
        self.m_dim = hidden_dim // 2
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Edge layers
        self.edge_norm1 = nn.LayerNorm(self.m_dim)
        self.edge_norm2 = nn.LayerNorm(1)

        self.edge_mlp1 = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + 1, self.hidden_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 2, self.m_dim),
            nn.SiLU(),
        )
        self.edge_mlp2 = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + 1, self.hidden_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 2, 1),
            nn.SiLU(),
        )

        # Node layers
        self.node_norm1 = nn.LayerNorm(self.hidden_dim)
        self.node_norm2 = nn.LayerNorm(self.hidden_dim)

        self.node_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim + self.m_dim, self.hidden_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )

    def forward(self, pos, x: Tensor, edge_index: Adj):
        vec = pos[edge_index[0]] - pos[edge_index[1]]
        dist = (vec**2).sum(dim=-1, keepdim=True)

        pos, x = self.propagate(
            edge_index,
            x=x,
            edge_attr=dist,
            pos=pos
            # edge_attr_topo=edge_attr, edge_attr_geom=dist, pos=pos,
        )

        return pos, x

    def message(self, pos_i, pos_j, x_i, x_j, edge_attr):
        m1_ij = self.edge_mlp1(torch.cat([x_i, x_j, edge_attr], dim=-1))
        m2_ij = self.edge_mlp2(torch.cat([x_i, x_j, edge_attr], dim=-1))
        m2_ij = self.edge_norm2(m2_ij)
        m2_ij = (pos_i - pos_j) * m2_ij
        return m1_ij, m2_ij

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        # get input tensors
        size = self.__check_input__(edge_index, size)
        coll_dict = self.__collect__(self.__user_args__, edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute("message", coll_dict)
        aggr_kwargs = self.inspector.distribute("aggregate", coll_dict)
        update_kwargs = self.inspector.distribute("update", coll_dict)

        # get messages
        m1_ij, m2_ij = self.message(**msg_kwargs)
        m1_ij = self.edge_norm1(m1_ij)

        # aggregate messages
        m1_i = self.aggregate(m1_ij, **aggr_kwargs)
        m2_i = self.aggregate(m2_ij, **aggr_kwargs)

        # get updated node features
        x = self.node_norm1(kwargs["x"])
        x = self.node_mlp(torch.cat([x, m1_i], dim=-1))
        x = self.node_norm2(x)
        x = kwargs["x"] + x

        pos = kwargs["pos"]
        pos = pos + m2_i
        return pos, x
