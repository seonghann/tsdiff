import torch
from torch import nn
from torch_scatter import scatter_add, scatter_mean
from torch_scatter import scatter
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj, dense_to_sparse, degree
from torch_geometric.nn import radius_graph
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


class EnsembleNetwork(nn.Module):
    def __init__(self, models):
        super().__init__()

        """
        edge_encoder:  Takes both edge type and edge length as input and outputs a vector
        [Note]: node embedding is done in SchNetEncoder
        """
        self.models = models
        self.config = models[0].config
        self.alphas = models[0].alphas
        self.betas = models[0].betas
        self.num_timesteps = models[0].num_timesteps

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
        out = self.models[0].forward(
                atom_type, 
                r_feat, 
                p_feat, 
                pos, 
                bond_index, 
                bond_type, 
                batch, 
                time_step, 
                return_edges=return_edges, 
                **kwargs
                )
        if return_edges:
             (
            edge_inv_global,
            edge_inv_local,
            edge_index_global,
            edge_index_local,
            edge_length_global,
            edge_length_local,
        ) = out
        else: 
            edge_inv_global, edge_inv_local = out
        
        #for model in self.models[1:]:
        for i, model in enumerate(self.models[1:]):
            out = model.forward(
                    atom_type, 
                    r_feat, 
                    p_feat, 
                    pos, 
                    bond_index, 
                    bond_type, 
                    batch, 
                    time_step, 
                    return_edges=return_edges, 
                    **kwargs
                    )
            eg, el = out[0], out[1]
            edge_inv_global += eg
            if edge_inv_local is not None and el is not None:
                edge_inv_local += el

        edge_inv_global /= len(self.models)
        if edge_inv_local is not None:
            edge_inv_local /= len(self.models)
        
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
                assert noise_from_time_t >= 0
                seq = range(denoise_from_time_t - n_steps, denoise_from_time_t)
                seq_next = [-1] + list(seq[:-1])
                noise = torch.randn(pos_init.size(), device=pos_init.device)

                alpha_t = self.alphas[denoise_from_time_t - 1]
                alpha_s = self.alphas[noise_from_time_t - 1] if noise_from_time_t != 0 else 1

                sigma = (1.0 - (alpha_t/alpha_s)) / alpha_t
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
            # for i, j in zip(reversed(seq), reversed(seq_next)):
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
                )
                if (
                    sampling_type == "generalized"
                    or sampling_type == "ddpm_noisy"
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

                elif sampling_type == "det":
                    step_size = step_lr * (sigmas[i] / 0.01) ** 2
                    pos_next = (
                        pos
                        + step_size * eps_pos / sigmas[i]
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
