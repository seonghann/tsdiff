import torch
from torch import nn
from torch_scatter import scatter_mean
import numpy as np

from tqdm.auto import tqdm
from utils.chem import BOND_TYPES
from models.geometry import eq_transform


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


class EnsembleSampler(nn.Module):
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
            edge_inv, edge_index, edge_length = out
        else:
            edge_inv = out

        # for model in self.models[1:]:
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
            edge_inv += out[0]

        edge_inv /= len(self.models)

        if return_edges:
            return edge_inv, edge_index, edge_length
        else:
            return edge_inv

    def dynamic_sampling(
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
        clip_pos=None,
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
        with torch.no_grad():
            # skip = self.num_timesteps // n_steps
            # seq = range(0, self.num_timesteps, skip)

            if noise_from_time_t is not None:
                assert denoise_from_time_t >= n_steps
                assert denoise_from_time_t >= noise_from_time_t
                assert noise_from_time_t >= 0
                seq = range(denoise_from_time_t - n_steps, denoise_from_time_t)
                seq_next = [-1] + list(seq[:-1])
                noise = torch.randn(pos_init.size(), device=pos_init.device)

                alpha_t = self.alphas[denoise_from_time_t - 1]
                alpha_s = self.alphas[noise_from_time_t - 1] if noise_from_time_t != 0 else 1

                sigma = (1.0 - (alpha_t / alpha_s)) / alpha_t
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

            for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), desc="sample"):
                t = torch.full(
                    size=(num_graphs,),
                    fill_value=i,
                    dtype=torch.long,
                    device=pos.device,
                )
                edge_inv, edge_index, edge_length = self(
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
                )  # (E, 1)

                node_eq = eq_transform(edge_inv, pos, edge_index, edge_length)
                eps_pos = clip_norm(node_eq, limit=clip)

                # Update
                sampling_type = kwargs.get("sampling_type", "ddpm")
                noise = torch.randn_like(pos)

                if sampling_type == "ddpm":
                    b = self.betas
                    t = t[0]
                    next_t = (torch.ones(1) * j).to(pos.device)
                    at = compute_alpha(b, t.long())
                    at_next = compute_alpha(b, next_t.long())
                    atm1 = at_next
                    beta_t = 1 - at / atm1
                    e = -eps_pos
                    pos_C = at.sqrt() * pos
                    pos0_from_e = (1.0 / at).sqrt() * pos_C - (
                        1.0 / at - 1
                    ).sqrt() * e
                    mean_eps = (
                        (atm1.sqrt() * beta_t) * pos0_from_e
                        + ((1 - beta_t).sqrt() * (1 - atm1)) * pos_C
                    ) / (1.0 - at)
                    mean = mean_eps
                    mask = 1 - (t == 0).float()
                    logvar = beta_t.log()

                    pos_next = (mean + mask * torch.exp(0.5 * logvar) * noise) / atm1.sqrt()

                elif sampling_type == "ld":
                    step_size = step_lr * (sigmas[i] / 0.01) ** 2
                    pos_next = (
                        pos
                        + step_size * eps_pos / sigmas[i]
                        + noise * torch.sqrt(step_size * 2)
                    )

                pos = pos_next

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


def center_pos(pos, batch):
    pos_center = pos - scatter_mean(pos, batch, dim=0)[batch]
    return pos_center


def clip_norm(vec, limit, p=2):
    norm = torch.norm(vec, dim=-1, p=2, keepdim=True)
    denom = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * denom
