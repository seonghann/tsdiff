import torch
from torch import nn
from torch_scatter import scatter_add, scatter_mean
from torch_scatter import scatter
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import numpy as np
from numpy import pi as PI
from tqdm.auto import tqdm

from utils.chem import BOND_TYPES
from utils import activation_loader
from ..common import (
    MultiLayerPerceptron,
    assemble_atom_pair_feature,
    generate_symmetric_edge_noise,
    extend_ts_graph_order_radius,
    index_set_subtraction,
)
from ..encoder import SchNetEncoder, GINEncoder, get_edge_encoder, load_encoder
from ..geometry import get_distance, get_angle, get_dihedral, eq_transform

# from diffusion import get_timestep_embedding, get_beta_schedule
import pdb


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


class DualEncoderEpsNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        """
        edge_encoder:  Takes both edge type and edge length as input and outputs a vector
        [Note]: node embedding is done in SchNetEncoder
        """
        self.edge_encoder_global = get_edge_encoder(config)
        self.edge_encoder_local = get_edge_encoder(config)
        assert config.hidden_dim % 2 == 0
        self.atom_embedding = nn.Embedding(100, config.hidden_dim // 2)
        self.atom_feat_embedding = nn.Linear(
            config.feat_dim, config.hidden_dim // 2, bias=False
        )

        """
        The graph neural network that extracts node-wise features.
        """
        print(config)
        self.encoder_global = load_encoder(config, "global_encoder")
        self.encoder_local = load_encoder(config, "local_encoder")

        """
        `output_mlp` takes a mixture of two nodewise features and edge features as input and outputs 
            gradients w.r.t. edge_length (out_dim = 1).
        """
        self.grad_global_dist_mlp = MultiLayerPerceptron(
            2 * config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, 1],
            activation=activation_loader(config.mlp_act),
        )

        self.grad_local_dist_mlp = MultiLayerPerceptron(
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
        self.model_global = nn.ModuleList(
            [self.edge_encoder_global, self.encoder_global, self.grad_global_dist_mlp]
        )
        self.model_local = nn.ModuleList(
            [self.edge_encoder_local, self.encoder_local, self.grad_local_dist_mlp]
        )

        self.model_type = config.type  # config.type  # 'diffusion'; 'dsm'

        if self.model_type == "diffusion":
            # denoising diffusion
            ## betas
            betas = get_beta_schedule(
                beta_schedule=config.beta_schedule,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                num_diffusion_timesteps=config.num_diffusion_timesteps,
            )
            betas = torch.from_numpy(betas).float()
            self.betas = nn.Parameter(betas, requires_grad=False)
            ## variances
            alphas = (1.0 - betas).cumprod(dim=0)
            self.alphas = nn.Parameter(alphas, requires_grad=False)
            self.num_timesteps = self.betas.size(0)
        elif self.model_type == "dsm":
            # denoising score matching
            sigmas = torch.tensor(
                np.exp(
                    np.linspace(
                        np.log(config.sigma_begin),
                        np.log(config.sigma_end),
                        config.num_noise_level,
                    )
                ),
                dtype=torch.float32,
            )
            self.sigmas = nn.Parameter(sigmas, requires_grad=False)  # (num_noise_level)
            self.num_timesteps = self.sigmas.size(0)  # betas.shape[0]

        if hasattr(config, "TS"):
            self.TS = config.TS
        else:
            self.TS = False

        from utils.chem import BOND_TYPES
        self.num_bond_types = len(BOND_TYPES)

        if self.TS:
            self.edge_cat_global = torch.nn.Sequential(
                torch.nn.Linear(
                    self.edge_encoder_global.out_channels * 2,
                    self.edge_encoder_global.out_channels,
                ),
                activation_loader(config.edge_cat_act),
                torch.nn.Linear(
                    self.edge_encoder_global.out_channels,
                    self.edge_encoder_global.out_channels,
                ),
            )
            self.edge_cat_local = torch.nn.Sequential(
                torch.nn.Linear(
                    self.edge_encoder_local.out_channels * 2,
                    self.edge_encoder_local.out_channels,
                ),
                activation_loader(config.edge_cat_act),
                torch.nn.Linear(
                    self.edge_encoder_local.out_channels,
                    self.edge_encoder_local.out_channels,
                ),
            )

    def _extend_condensed_graph_edge(self,N, pos, bond_index, bond_type, batch, cutoff=None, edge_order=None):

        if cutoff is None: cutoff = self.config.cutoff
        if edge_order is None: edge_order = self.config.edge_order

        out = extend_ts_graph_order_radius(
            num_nodes=N,
            pos=pos,
            edge_index=bond_index,
            edge_type=bond_type,
            batch=batch,
            order=edge_order,
            cutoff=cutoff,
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
    
    def _condensed_edge_embedding(
            self, 
            edge_length, 
            edge_type_r, 
            edge_type_p, 
            edge_type="global", 
            edge_attr=None,
            emb_type="bond_w_d"):

        assert emb_type in ["bond_w_d", "bond_wo_d", "add_d"]

        encoder = getattr(self, f"edge_encoder_{edge_type}")
        cat_fn = getattr(self, f"edge_cat_{edge_type}")

        if emb_type == "bond_wo_d":
            edge_attr_r = encoder.bond_emb(edge_type_r)
            edge_attr_p = encoder.bond_emb(edge_type_p)
            edge_attr = cat_fn(torch.cat([edge_attr_r, edge_attr_p], dim=-1))

        elif emb_type == "bond_w_d":
            edge_attr_r = encoder(edge_length=edge_length, edge_type=edge_type_r)  # Embed edges
            edge_attr_p = encoder(edge_length=edge_length, edge_type=edge_type_p)
            edge_attr = cat_fn(torch.cat([edge_attr_r, edge_attr_p], dim=-1))

        elif emb_type == "add_d":
            edge_attr = encoder.mlp(edge_length) * edge_attr
        return edge_attr
    
    def forward_(self, atom_type, r_feat, p_feat, pos, bond_index, bond_type, batch,
            enc_type="global", **kwargs):
        """
        Args:
            atom_type:  Types of atoms, (N, ).
            bond_index: Indices of bonds (not extended, not radius-graph), (2, E).
            bond_type:  Bond types, (E, ).
            batch:      Node index to graph index, (N, ).
        """
        N = atom_type.size(0)
        # --------------------------------------------------------------------
        # condensed atom embedding
        atom_emb = self.atom_embedding(atom_type)
        atom_feat_emb_r = self.atom_feat_embedding(r_feat.float())
        atom_feat_emb_p = self.atom_feat_embedding(p_feat.float())
        z = torch.cat(
            [atom_emb + atom_feat_emb_r, atom_feat_emb_p - atom_feat_emb_r], dim=-1
        )
        
        if enc_type == "global":
            edge_order = self.config.global_edge_order
            edge_cutoff = self.config.global_edge_cutoff
            edge_order_pred = self.config.global_pred_edge_order
            
        elif enc_type == "local":
            edge_order = self.config.local_edge_order
            edge_cutoff = self.config.local_edge_cutoff
            edge_order_pred = self.config.local_pred_edge_order
        else:
            raise
            
        # --------------------------------------------------------------------
        # edge extension
        (
                edge_index, 
                _, 
                edge_type_r, 
                edge_type_p
                ) = self._extend_condensed_graph_edge(
                        N, 
                        pos, 
                        bond_index, 
                        bond_type, 
                        batch,
                        edge_order=edge_order,
                        cutoff=edge_cutoff
                        )
        edge_length= get_distance(pos, edge_index).unsqueeze(-1)  # (E, 1)

        # --------------------------------------------------------------------
        # global edge embedding
        edge_attr_wo_length = False
        edge_attr= self._condensed_edge_embedding(
                edge_length, 
                edge_type_r, 
                edge_type_p, 
                edge_type=enc_type,
                emb_type="bond_wo_d" if edge_attr_wo_length else "bond_w_d",
                )
        sigma_edge = 1.0
        
        # --------------------------------------------------------------------
        # global encoding and pair encoding
        enc = getattr(self, f"encoder_{enc_type}")
        node_attr= enc(
                z=z,
                edge_index=edge_index,
                edge_length=edge_length,
                pos=pos,
                edge_attr=edge_attr,
                edge_type=(edge_type_r, edge_type_p),
                embed_node=False,
                )

        if edge_order_pred != edge_order:
            (
                    edge_index, 
                    _, 
                    edge_type_r, 
                    edge_type_p
                    ) = self._extend_condensed_graph_edge(
                            N, 
                            pos, 
                            bond_index, 
                            bond_type, 
                            batch,
                            edge_order=edge_order_pred,
                            cutoff=edge_cutoff
                            )
            edge_length= get_distance(pos, edge_index).unsqueeze(-1)  # (E, 1)
            edge_attr= self._condensed_edge_embedding(
                    edge_length, 
                    edge_type_r, 
                    edge_type_p, 
                    edge_type=enc_type,
                    emb_type="bond_w_d",
                    )
        
        else:
            if edge_attr_wo_length:
                edge_attr= self._condensed_edge_embedding(
                        edge_length,
                        None,
                        None,
                        edge_attr=edge_attr,
                        edge_type=enc_type,
                        emb_type="add_d"
                        )

        h_pair= assemble_atom_pair_feature(
            node_attr=node_attr,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )  # (E_global, 2H)
        
        ## Invariant features of edges (radius graph, global)
        grad_mlp = getattr(self, f"grad_{enc_type}_dist_mlp")
        edge_inv= grad_mlp(h_pair)  # (E_global, 1)
        # edge_inv_global /= sigma_edge

        return edge_inv, edge_index, edge_length,

    def forward(
        self,
        atom_type,
        r_feat,
        p_feat,
        pos,
        bond_index,
        bond_type,
        batch,
        time_step,
        return_edges=True,
        **kwargs,
    ):
        """
        Args:
            atom_type:  Types of atoms, (N, ).
            bond_index: Indices of bonds (not extended, not radius-graph), (2, E).
            bond_type:  Bond types, (E, ).
            batch:      Node index to graph index, (N, ).
        """
        N = atom_type.size(0)
        out_global = self.forward_(
                atom_type, 
                r_feat, 
                p_feat, 
                pos, 
                bond_index, 
                bond_type, 
                batch,
                enc_type="global", 
                )

        edge_inv_global, edge_index_global, edge_length_global = out_global

        if self.config.dual_encoding:
            # --------------------------------------------------------------------
            # local edge extension
            assert self.config.local_edge_cutoff < 0.1
            out_local = self.forward_(
                    atom_type, 
                    r_feat, 
                    p_feat, 
                    pos, 
                    bond_index, 
                    bond_type, 
                    batch,
                    enc_type="local", 
                    )
            
            edge_inv_local, edge_index_local, edge_length_local = out_local
            
            # edge_inv_global = ...
            # edge_index_global = ...
            idx_ = index_set_subtraction(
                        edge_index_global, 
                        edge_index_local, 
                        max_num_nodes=N
                        )
            edge_index_global = edge_index_global[:, idx_]
            edge_inv_global = edge_inv_global[idx_]
            edge_length_global = edge_length_global[idx_]
            
        else:
            edge_length_local = None
            edge_inv_local = None
            edge_index_local = None
        
        if return_edges:
            return (
                edge_inv_global,
                edge_inv_local,
                edge_index_global,
                edge_index_local,
                edge_length_global,
                edge_length_local,
            )
        else:
            return edge_inv_global, edge_inv_local

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
        return_unreduced_loss=False,
        return_unreduced_edge_loss=False,
        extend_order=True,
        extend_radius=True,
        is_sidechain=None,
    ):
        if self.model_type == "diffusion":
            return self.get_loss_diffusion(
                atom_type,
                r_feat,
                p_feat,
                pos,
                bond_index,
                bond_type,
                batch,
                num_nodes_per_graph,
                num_graphs,
                anneal_power,
                return_unreduced_loss,
                return_unreduced_edge_loss,
                extend_order,
                extend_radius,
                is_sidechain,
            )
    
    def get_loss_diffusion(
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
        return_unreduced_loss=False,
        return_unreduced_edge_loss=False,
        extend_order=True,
        extend_radius=True,
        is_sidechain=None,
    ):
        N = atom_type.size(0)
        node2graph = batch
        # Four elements for DDPM: original_data(pos), gaussian_noise(pos_noise), beta(sigma), time_step
        # Sample noise levels
        t0 = 0 if not hasattr(self.config, "t0") else self.config.t0
        t1 = self.num_timesteps if not hasattr(self.config, "t1") else self.config.t1

        time_step = torch.randint(
            t0, t1, size=(num_graphs // 2 + 1,), device=pos.device
        )
        time_step = torch.cat([time_step, t0+t1-1-time_step], dim=0)[:num_graphs]
        #time_step = torch.randint(
        #    0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=pos.device
        #)
        #time_step = torch.cat([time_step, self.num_timesteps - time_step - 1], dim=0)[
        #    :num_graphs
        #]
        a = self.alphas.index_select(0, time_step)  # (G, )
        # Perterb pos
        a_pos = a.index_select(0, node2graph).unsqueeze(-1)  # (N, 1)
        pos_noise = torch.zeros(size=pos.size(), device=pos.device)
        pos_noise.normal_()
        pos_perturbed = pos + pos_noise * (1.0 - a_pos).sqrt() / a_pos.sqrt()

        # Update invariant edge features, as shown in equation 5-7
        (
            edge_inv_global,
            edge_inv_local,
            edge_index_global,
            edge_index_local,
            edge_length_global,
            edge_length_local,
        ) = self(
            atom_type=atom_type,
            r_feat=r_feat,
            p_feat=p_feat,
            pos=pos_perturbed,
            bond_index=bond_index,
            bond_type=bond_type,
            batch=batch,
            time_step=time_step,
            return_edges=True,
            extend_order=extend_order,
            extend_radius=extend_radius,
            is_sidechain=is_sidechain,
        )  # (E_global, 1), (E_local, 1)

        # calculate global
        # ----------------------------------------------------------------
        # setting for global
        edge_index = edge_index_global
        edge2graph = node2graph.index_select(0, edge_index[0])
        a_edge = a.index_select(0, edge2graph).unsqueeze(-1)  # (E, 1)

        # compute original and perturbed distances
        d_gt = get_distance(pos, edge_index).unsqueeze(-1)  # (E, 1)
        d_perturbed = edge_length_global

        d_target = (
            (d_gt - d_perturbed) / (1.0 - a_edge).sqrt() * a_edge.sqrt()
        )  # (E_global, 1), denoising direction

        # re-parametrization, distance to position
        target_d_global = d_target
        target_pos_global = eq_transform(
            target_d_global, pos_perturbed, edge_index, edge_length_global
        )
        node_eq_global = eq_transform(
            edge_inv_global, pos_perturbed, edge_index, edge_length_global
        )

        # calc loss
        loss_global = (node_eq_global - target_pos_global) ** 2
        loss_global = torch.sum(loss_global, dim=-1, keepdim=True)
        loss = loss_global

        if self.config.dual_encoding:
            # calculate local
            # ----------------------------------------------------------------
            # setting for local
            edge_index = edge_index_local
            edge2graph = node2graph.index_select(0, edge_index[0])
            a_edge = a.index_select(0, edge2graph).unsqueeze(-1)  # (E, 1)

            # compute original and perturbed distances
            d_gt = get_distance(pos, edge_index).unsqueeze(-1)  # (E, 1)
            d_perturbed = edge_length_local
            d_target = (
                (d_gt - d_perturbed) / (1.0 - a_edge).sqrt() * a_edge.sqrt()
            )  # (E_local, 1), denoising direction

            # re-parametrization, distance to position
            target_d_local = d_target
            target_pos_local = eq_transform(
                target_d_local, pos_perturbed, edge_index, edge_length_local
            )
            node_eq_local = eq_transform(
                edge_inv_local, pos_perturbed, edge_index, edge_length_local
            )

            # calc loss
            loss_local = (node_eq_local - target_pos_local) ** 2
            loss_local = torch.sum(loss_local, dim=-1, keepdim=True)
            loss = loss + loss_local

        else:
            loss_local = loss_global * 0

        if return_unreduced_edge_loss:
            pass
        elif return_unreduced_loss:
            return loss, loss_global, loss_local
        else:
            return loss

    def langevin_dynamics_sample(
        self,
        atom_type,
        r_feat,
        p_feat,
        pos_init,
        bond_index,
        bond_type,
        batch,
        num_graphs,
        extend_order,
        extend_radius=True,
        n_steps=100,
        step_lr=0.0000010,
        clip=1000,
        clip_local=None,
        clip_pos=None,
        min_sigma=0,
        is_sidechain=None,
        global_start_sigma=float("inf"),
        w_global=0.2,
        w_reg=1.0,
        denoise_from_time_t=None,
        noise_from_time_t=None,
        **kwargs,
    ):
        if self.model_type == "diffusion":
            return self.langevin_dynamics_sample_diffusion(
                atom_type,
                r_feat,
                p_feat,
                pos_init,
                bond_index,
                bond_type,
                batch,
                num_graphs,
                extend_order,
                extend_radius,
                n_steps,
                step_lr,
                clip,
                clip_local,
                clip_pos,
                min_sigma,
                is_sidechain,
                global_start_sigma,
                w_global,
                w_reg,
                sampling_type=kwargs.get("sampling_type", "ddpm_noisy"),
                eta=kwargs.get("eta", 1.0),
                denoise_from_time_t=denoise_from_time_t,
                noise_from_time_t=noise_from_time_t,
            )

    def langevin_dynamics_sample_diffusion(
        self,
        atom_type,
        r_feat,
        p_feat,
        pos_init,
        bond_index,
        bond_type,
        batch,
        num_graphs,
        extend_order,
        extend_radius=True,
        n_steps=100,
        step_lr=0.0000010,
        clip=1000,
        clip_local=None,
        clip_pos=None,
        min_sigma=0,
        is_sidechain=None,
        global_start_sigma=float("inf"),
        w_global=0.2,
        w_reg=1.0,
        denoise_from_time_t=None,
        noise_from_time_t=None,
        **kwargs,
    ):
        def compute_alpha(beta, t):
            beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
            a = (1 - beta).cumprod(dim=0).index_select(0, t + 1)  # .view(-1, 1, 1, 1)
            return a

        sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()
        pos_traj = []
        if is_sidechain is not None:
            assert pos_gt is not None, "need crd of backbone for sidechain prediction"
        with torch.no_grad():
            # skip = self.num_timesteps // n_steps
            # seq = range(0, self.num_timesteps, skip)

            ## to test sampling with less intermediate diffusion steps
            # n_steps: the num of steps
            if noise_from_time_t is not None:
                assert denoise_from_time_t >= n_steps
                assert denoise_from_time_t >= noise_from_time_t
                assert noise_from_time_t >= 1
                seq = range(denoise_from_time_t - n_steps, denoise_from_time_t)
                seq_next = [-1] + list(seq[:-1])
                noise = torch.randn(pos_init.size(), device=pos_init.device)
                sigma = 1.0 - (
                    self.alphas[denoise_from_time_t - 1]
                    / self.alphas[noise_from_time_t - 1]
                )
                sigma /= self.alphas[denoise_from_time_t - 1]
                sigma = sigma.sqrt()
                pos = pos_init + noise * sigma

                print(
                    f"Noising from t={noise_from_time_t} to t={denoise_from_time_t}\n"
                    f"Denoise from t={denoise_from_time_t} to t={denoise_from_time_t-n_steps}"
                )

            elif denoise_from_time_t is not None:
                assert denoise_from_time_t >= n_steps
                seq = range(denoise_from_time_t - n_steps, denoise_from_time_t)
                seq_next = [-1] + list(seq[:-1])
                pos = pos_init
                print(
                    f"Start with zero-noise\n"
                    f"Denoise from t={denoise_from_time_t} to t={denoise_from_time_t-n_steps}"
                )

            else:
                seq = range(self.num_timesteps - n_steps, self.num_timesteps)
                seq_next = [-1] + list(seq[:-1])
                pos = pos_init * sigmas[-1]

            print("Initial Position")
            print(pos)
            print(pos.shape)

            if is_sidechain is not None:
                pos[~is_sidechain] = pos_gt[~is_sidechain]
            for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), desc="sample"):
                t = torch.full(
                    size=(num_graphs,),
                    fill_value=i,
                    dtype=torch.long,
                    device=pos.device,
                )

                (
                    edge_inv_global,
                    edge_inv_local,
                    edge_index_global,
                    edge_index_local,
                    edge_length_global,
                    edge_length_local,
                ) = self(
                    atom_type=atom_type,
                    r_feat=r_feat,
                    p_feat=p_feat,
                    pos=pos,
                    bond_index=bond_index,
                    bond_type=bond_type,
                    batch=batch,
                    time_step=t,
                    return_edges=True,
                    extend_order=extend_order,
                    extend_radius=extend_radius,
                    is_sidechain=is_sidechain,
                )  # (E_global, 1), (E_local, 1)

                # Global
                # if sigmas[i] < global_start_sigma:
                edge_inv_global = edge_inv_global
                node_eq_global = eq_transform(
                    edge_inv_global, pos, edge_index_global, edge_length_global
                )
                node_eq_global = clip_norm(node_eq_global, limit=clip)
                
                # Local
                if self.config.dual_encoding:
                    node_eq_local = eq_transform(
                        edge_inv_local, pos, edge_index_local, edge_length_local,
                    )
                    if clip_local is not None:
                        node_eq_local = clip_norm(node_eq_local, limit=clip_local)
                else:
                    node_eq_local = 0.0

                # Sum
                eps_pos = (node_eq_local + node_eq_global * w_global)  # + eps_pos_reg * w_reg
                # eps_pos = node_eq_local * (1 - w_global) + node_eq_global * w_global # + eps_pos_reg * w_reg

                # Update

                sampling_type = kwargs.get(
                    "sampling_type", "ddpm_noisy"
                )  # types: generalized, ddpm_noisy, ld

                noise = torch.randn_like(
                    pos
                )  #  center_pos(torch.randn_like(pos), batch)
                if (
                    sampling_type == "generalized"
                    or sampling_type == "ddpm_noisy"
                    or sampling_type == "ddpm_det"
                ):
                    b = self.betas
                    t = t[0]
                    next_t = (torch.ones(1) * j).to(pos.device)
                    at = compute_alpha(b, t.long())
                    at_next = compute_alpha(b, next_t.long())
                    if sampling_type == "generalized":
                        eta = kwargs.get("eta", 1.0)
                        et = -eps_pos
                        ## original
                        # pos0_t = (pos - et * (1 - at).sqrt()) / at.sqrt()
                        ## reweighted
                        # pos0_t = pos - et * (1 - at).sqrt() / at.sqrt()
                        c1 = (
                            eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                        )
                        c2 = ((1 - at_next) - c1**2).sqrt()
                        # pos_next = at_next.sqrt() * pos0_t + c1 * noise + c2 * et
                        # pos_next = pos0_t + c1 * noise / at_next.sqrt() + c2 * et / at_next.sqrt()

                        # pos_next = pos + et * (c2 / at_next.sqrt() - (1 - at).sqrt() / at.sqrt()) + noise * c1 / at_next.sqrt()
                        step_size_pos_ld = step_lr * (sigmas[i] / 0.01) ** 2 / sigmas[i]
                        step_size_pos_generalized = 5 * (
                            (1 - at).sqrt() / at.sqrt() - c2 / at_next.sqrt()
                        )
                        step_size_pos = (
                            step_size_pos_ld
                            if step_size_pos_ld < step_size_pos_generalized
                            else step_size_pos_generalized
                        )

                        step_size_noise_ld = torch.sqrt(
                            (step_lr * (sigmas[i] / 0.01) ** 2) * 2
                        )
                        step_size_noise_generalized = 3 * (c1 / at_next.sqrt())
                        step_size_noise = (
                            step_size_noise_ld
                            if step_size_noise_ld < step_size_noise_generalized
                            else step_size_noise_generalized
                        )

                        pos_next = pos - et * step_size_pos + noise * step_size_noise

                    elif sampling_type == "ddpm_det":
                        atm1 = at_next
                        beta_t = 1 - at / atm1
                        e = -eps_pos
                        pos0_from_e = (1.0 / at).sqrt() * pos - (
                            1.0 / at - 1
                        ).sqrt() * e
                        mean_eps = (
                            (atm1.sqrt() * beta_t) * pos0_from_e
                            + ((1 - beta_t).sqrt() * (1 - atm1)) * pos
                        ) / (1.0 - at)
                        mean = mean_eps
                        mask = 1 - (t == 0).float()
                        logvar = (beta_t * (1 - atm1) / (1 - at)).log()

                        pos_next = mean + mask * torch.exp(0.5 * logvar) * noise

                    elif sampling_type == "ddpm_noisy":
                        atm1 = at_next
                        beta_t = 1 - at / atm1
                        e = -eps_pos
                        pos0_from_e = (1.0 / at).sqrt() * pos - (
                            1.0 / at - 1
                        ).sqrt() * e
                        mean_eps = (
                            (atm1.sqrt() * beta_t) * pos0_from_e
                            + ((1 - beta_t).sqrt() * (1 - atm1)) * pos
                        ) / (1.0 - at)
                        mean = mean_eps
                        mask = 1 - (t == 0).float()
                        logvar = beta_t.log()

                        pos_next = mean + mask * torch.exp(0.5 * logvar) * noise

                elif sampling_type == "ld":
                    step_size = step_lr * (sigmas[i] / 0.01) ** 2
                    pos_next = (
                        pos
                        + step_size * eps_pos / sigmas[i]
                        + noise * torch.sqrt(step_size * 2)
                    )

                pos = pos_next

                if is_sidechain is not None:
                    pos[~is_sidechain] = pos_gt[~is_sidechain]

                if torch.isnan(pos).any():
                    print("NaN detected. Please restart.")
                    raise FloatingPointError()
                pos = center_pos(pos, batch)
                if clip_pos is not None:
                    pos = torch.clamp(pos, min=-clip_pos, max=clip_pos)
                pos_traj.append(pos.clone().cpu())
            print("Generated Position")
            print(pos)
        return pos, pos_traj


def is_bond(edge_type):
    return torch.logical_and(edge_type < len(BOND_TYPES), edge_type > 0)


def is_angle_edge(edge_type):
    return edge_type == len(BOND_TYPES) + 1 - 1


def is_dihedral_edge(edge_type):
    return edge_type == len(BOND_TYPES) + 2 - 1


def is_radius_edge(edge_type):
    return edge_type == 0


def is_local_edge(edge_type):
    return edge_type > 0


def calc_local_edge_mask(edge_index_global, edge_index_local):
    adj_g = to_dense_adj(edge_index_global)
    adj_l = to_dense_adj(edge_index_local)
    adj = torch.where(adj_l != 0, -adj_l, adj_g)
    idx, v = dense_to_sparse(adj)
    return v == -1


def is_train_edge(edge_index, is_sidechain):
    if is_sidechain is None:
        return torch.ones(edge_index.size(1), device=edge_index.device).bool()
    else:
        is_sidechain = is_sidechain.bool()
        return torch.logical_or(
            is_sidechain[edge_index[0]], is_sidechain[edge_index[1]]
        )


def regularize_bond_length(edge_type, edge_length, rng=5.0):
    mask = is_bond(edge_type).float().reshape(-1, 1)
    d = -torch.clamp(edge_length - rng, min=0.0, max=float("inf")) * mask
    return d


def center_pos(pos, batch):
    pos_center = pos - scatter_mean(pos, batch, dim=0)[batch]
    return pos_center


def clip_norm(vec, limit, p=2):
    norm = torch.norm(vec, dim=-1, p=2, keepdim=True)
    denom = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * denom
