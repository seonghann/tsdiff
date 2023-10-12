import torch
from torch import nn
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import numpy as np

from utils.chem import BOND_TYPES
from utils import activation_loader
from ..common import MultiLayerPerceptron, assemble_atom_pair_feature, extend_ts_graph_order_radius
from ..encoder import get_edge_encoder, load_encoder
from ..geometry import get_distance, eq_transform


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


# class DualEncoderEpsNetwork(nn.Module):
class CondenseEncoderEpsNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        """
        edge_encoder:  Takes both edge type and edge length as input and outputs a vector
        [Note]: node embedding is done in SchNetEncoder
        """
        self.edge_encoder = get_edge_encoder(config)
        assert config.hidden_dim % 2 == 0
        self.atom_embedding = nn.Embedding(100, config.hidden_dim // 2)
        self.atom_feat_embedding = nn.Linear(
            config.feat_dim, config.hidden_dim // 2, bias=False
        )

        """
        The graph neural network that extracts node-wise features.
        """
        self.encoder = load_encoder(config, "encoder")

        """
        `output_mlp` takes a mixture of two nodewise features and edge features as input and outputs
            gradients w.r.t. edge_length (out_dim = 1).
        """
        self.grad_dist_mlp = MultiLayerPerceptron(
            2 * config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, 1],
            activation=activation_loader(config.mlp_act),
        )

        """
        Incorporate parameters together
        """
        self.model_embedding = nn.ModuleList(
            [
                self.atom_embedding,
                self.atom_feat_embedding,
            ]
        )
        self.model = nn.ModuleList(
            [self.edge_encoder, self.encoder, self.grad_dist_mlp]
        )

        betas = get_beta_schedule(
            beta_schedule=config.beta_schedule,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            num_diffusion_timesteps=config.num_diffusion_timesteps,
        )
        betas = torch.from_numpy(betas).float()
        self.betas = nn.Parameter(betas, requires_grad=False)
        # variances
        alphas = (1.0 - betas).cumprod(dim=0)
        self.alphas = nn.Parameter(alphas, requires_grad=False)
        self.num_timesteps = self.betas.size(0)

        self.num_bond_types = len(BOND_TYPES)
        self.edge_cat = torch.nn.Sequential(
            torch.nn.Linear(
                self.edge_encoder.out_channels * 2,
                self.edge_encoder.out_channels,
            ),
            activation_loader(config.edge_cat_act),
            torch.nn.Linear(
                self.edge_encoder.out_channels,
                self.edge_encoder.out_channels,
            ),
        )

    def _extend_condensed_graph_edge(self, pos, bond_index, bond_type, batch, **kwargs):
        N = pos.size(0)
        cutoff = kwargs.get("cutoff", self.config.edge_cutoff)
        edge_order = kwargs.get("edge_order", self.config.edge_order)

        _g_ext = extend_ts_graph_order_radius
        out = _g_ext(
            N, pos, bond_index, bond_type, batch, order=edge_order, cutoff=cutoff
        )
        edge_index_global, edge_index_local, edge_type_r, edge_type_p = out
        # local index             : (i, j) pairs which are edge of R or P.
        # edge_type_r/edge_type_p : 0, 1, 2, ... 23, 24, ...
        #                           0 -> no edge (bond)
        #                           1, 2, 3 ..-> bond type
        #                           23, 24 -> meaning no bond, but higher order edge. (2-hop or 3-hop)
        # global index            : atom pairs (i, j) which are closer than cutoff
        #                           are added to local_index.
        #

        edge_type_global = torch.zeros_like(edge_index_global[0]) - 1
        adj_global = to_dense_adj(
            edge_index_global, edge_attr=edge_type_global, max_num_nodes=N
        )
        adj_local_r = to_dense_adj(
            edge_index_local, edge_attr=edge_type_r, max_num_nodes=N
        )
        adj_local_p = to_dense_adj(
            edge_index_local, edge_attr=edge_type_p, max_num_nodes=N
        )
        adj_global_r = torch.where(adj_local_r != 0, adj_local_r, adj_global)
        adj_global_p = torch.where(adj_local_p != 0, adj_local_p, adj_global)
        edge_index_global_r, edge_type_global_r = dense_to_sparse(adj_global_r)
        edge_index_global_p, edge_type_global_p = dense_to_sparse(adj_global_p)
        edge_type_global_r[edge_type_global_r < 0] = 0
        edge_type_global_p[edge_type_global_p < 0] = 0
        edge_index_global = edge_index_global_r

        return edge_index_global, edge_index_local, edge_type_global_r, edge_type_global_p

    def _condensed_edge_embedding(self, edge_length, edge_type_r, edge_type_p,
                                  edge_attr=None, emb_type="bond_w_d"):

        assert emb_type in ["bond_w_d", "bond_wo_d", "add_d"]
        _enc = self.edge_encoder
        _cat_fn = self.edge_cat

        if emb_type == "bond_wo_d":
            edge_attr_r = _enc.bond_emb(edge_type_r)
            edge_attr_p = _enc.bond_emb(edge_type_p)
            edge_attr = _cat_fn(torch.cat([edge_attr_r, edge_attr_p], dim=-1))

        elif emb_type == "bond_w_d":
            edge_attr_r = _enc(edge_length, edge_type_r)  # Embed edges
            edge_attr_p = _enc(edge_length, edge_type_p)
            edge_attr = _cat_fn(torch.cat([edge_attr_r, edge_attr_p], dim=-1))

        elif emb_type == "add_d":
            edge_attr = _enc.mlp(edge_length) * edge_attr

        return edge_attr

    def forward_(self, atom_type, r_feat, p_feat, pos, bond_index, bond_type, batch, **kwargs):
        """
        Args:
            atom_type:  Types of atoms, (N, ).
            bond_index: Indices of bonds (not extended, not radius-graph), (2, E).
            bond_type:  Bond types, (E, ).
            batch:      Node index to graph index, (N, ).
        """
        _g_ext = self._extend_condensed_graph_edge
        _e_emb = self._condensed_edge_embedding
        _a_emb = self.atom_embedding
        _af_emb = self.atom_feat_embedding
        _enc = self.encoder

        # condensed atom embedding
        a_emb = _a_emb(atom_type)
        af_emb_r = _af_emb(r_feat.float())
        af_emb_p = _af_emb(p_feat.float())
        z1 = a_emb + af_emb_r
        z2 = af_emb_p - af_emb_r
        z = torch.cat([z1, z2], dim=-1)

        # edge extension
        edge_index, _, edge_type_r, edge_type_p = _g_ext(
            pos,
            bond_index,
            bond_type,
            batch,
        )
        edge_length = get_distance(pos, edge_index).unsqueeze(-1)  # (E, 1)

        # edge embedding
        edge_attr = _e_emb(
            edge_length,
            edge_type_r,
            edge_type_p,
        )

        # encoding TS geometric graph and atom-pair
        node_attr = _enc(z, edge_index, edge_length, edge_attr=edge_attr)

        edge_ord4inp = self.config.edge_order
        edge_ord4out = self.config.pred_edge_order
        if edge_ord4inp != edge_ord4out:
            edge_index, _, edge_type_r, edge_type_p = _g_ext(
                pos,
                bond_index,
                bond_type,
                batch,
                edge_order=edge_ord4out,
            )
            edge_length = get_distance(pos, edge_index).unsqueeze(-1)  # (E, 1)
            edge_attr = _e_emb(
                edge_length,
                edge_type_r,
                edge_type_p,
            )

        h_pair = assemble_atom_pair_feature(node_attr, edge_index, edge_attr)  # (E, 2H)
        edge_inv = self.grad_dist_mlp(h_pair)  # (E, 1)

        return edge_inv, edge_index, edge_length

    def forward(self, atom_type, r_feat, p_feat, pos, bond_index, bond_type, batch,
                time_step, return_edges=True, **kwargs):
        """
        Args:
            atom_type:  Types of atoms, (N, ).
            bond_index: Indices of bonds (not extended, not radius-graph), (2, E).
            bond_type:  Bond types, (E, ).
            batch:      Node index to graph index, (N, ).
        """
        out = self.forward_(
            atom_type,
            r_feat,
            p_feat,
            pos,
            bond_index,
            bond_type,
            batch,
        )

        edge_inv, edge_index, edge_length = out

        if return_edges:
            return edge_inv, edge_index, edge_length
        else:
            return edge_inv

    def get_loss(
        self,
        atom_type,
        r_feat,
        p_feat,
        pos,
        bond_index,
        bond_type,
        batch,
        num_nodes_per_graph,
        num_graphs,
        anneal_power=2.0,
        extend_order=True,
        extend_radius=True,
    ):
        node2graph = batch
        dev = pos.device
        # set time step and noise level
        t0 = self.config.get("t0", 0)
        t1 = self.config.get("t1", self.num_timesteps)

        sz = num_graphs // 2 + 1
        half_1 = torch.randint(t0, t1, size=(sz,), device=dev)
        half_2 = t0 + t1 - 1 - half_1
        time_step = torch.cat([half_1, half_2], dim=0)[:num_graphs]
        a = self.alphas.index_select(0, time_step)  # (G, )

        # Perterb pos
        a_pos = a.index_select(0, node2graph).unsqueeze(-1)  # (N, 1)
        pos_noise = torch.randn(size=pos.size(), device=dev)
        pos_perturbed = pos + pos_noise * (1.0 - a_pos).sqrt() / a_pos.sqrt()

        # prediction
        edge_inv, edge_index, edge_length = self(
            atom_type, r_feat, p_feat, pos_perturbed, bond_index, bond_type,
            batch, time_step, return_edges=True, extend_order=extend_order,
            extend_radius=extend_radius,
        )  # (E, 1)
        node_eq = eq_transform(
            edge_inv, pos_perturbed, edge_index, edge_length
        )  # chain rule (re-parametrization, distance to position)

        # setup for target
        edge2graph = node2graph.index_select(0, edge_index[0])
        a_edge = a.index_select(0, edge2graph).unsqueeze(-1)  # (E, 1)

        # compute original and perturbed distances
        d_gt = get_distance(pos, edge_index).unsqueeze(-1)  # (E, 1)
        d_perturbed = edge_length

        # compute target
        d_target = d_gt - d_perturbed  # (E, 1), denoising direction
        d_target = d_target / (1.0 - a_edge).sqrt() * a_edge.sqrt()
        pos_target = eq_transform(
            d_target, pos_perturbed, edge_index, edge_length
        )  # chain rule (re-parametrization, distance to position)

        # calc loss
        loss = (node_eq - pos_target) ** 2
        loss = torch.sum(loss, dim=-1, keepdim=True)

        return loss
