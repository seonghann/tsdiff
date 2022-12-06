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
from ..common import MultiLayerPerceptron, assemble_atom_pair_feature, generate_symmetric_edge_noise, extend_ts_graph_order_radius
from ..encoder import EGNNMixed2DEncoder, get_edge_encoder
from ..geometry import get_distance, get_angle, get_dihedral, eq_transform
# from diffusion import get_timestep_embedding, get_beta_schedule
import pdb


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
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


class Mixed2DEpsNetwork(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        """
        edge_encoder:  Takes both edge type and edge length as input and outputs a vector
        [Note]: node embedding is done in SchNetEncoder
        """
        self.bond_emb = torch.nn.Embedding(100, config.hidden_dim)
        assert config.hidden_dim % 2 == 0
        self.atom_embedding = nn.Embedding(100, config.hidden_dim//2)
        self.atom_feat_embedding = nn.Linear(config.feat_dim ,config.hidden_dim//2, bias=False)
        
        """
        The graph neural network that extracts node-wise features.
        """
        self.graph_encoder = EGNNMixed2DEncoder(
            hidden_dim=config.hidden_dim,
            num_convs=config.num_convs,
            dropout=config.dropout,
        )

        """
        `output_mlp` takes a mixture of two nodewise features and edge features as input and outputs 
            gradients w.r.t. edge_length (out_dim = 1).
        """
        self.grad_dist_mlp = MultiLayerPerceptron(
            2 * config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, 1], 
            activation=activation_loader(config.mlp_act)
        )

        '''
        Incorporate parameters together
        '''
        self.model_params = nn.ModuleList([
            self.atom_embedding, 
            self.atom_feat_embedding, 
            self.bond_emb, 
            self.graph_encoder, 
            self.grad_dist_mlp
            ])
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
        alphas = (1. - betas).cumprod(dim=0)
        self.alphas = nn.Parameter(alphas, requires_grad=False)
        self.num_timesteps = self.betas.size(0)
    

    def forward(self, atom_type, r_feat, p_feat, pos, bond_index, bond_type, batch, time_step, 
                return_edges=False, **kwargs):
        """
        Args:
            atom_type:  Types of atoms, (N, ).
            bond_index: Indices of bonds (not extended, not radius-graph), (2, E).
            bond_type:  Bond types, (E, ).
            batch:      Node index to graph index, (N, ).
        """
        N = atom_type.size(0)
        out = extend_ts_graph_order_radius( 
                    num_nodes=N, 
                    pos=pos, 
                    edge_index=bond_index, 
                    edge_type=bond_type, 
                    batch=batch, 
                    order=self.config.edge_order, 
                    cutoff=self.config.cutoff,
                    )
        edge_index_global, edge_index_local, edge_type_r, edge_type_p = out

        edge_type_global = torch.zeros_like(edge_index_global[0]) - 1
        adj_global = to_dense_adj(edge_index_global, edge_attr=edge_type_global, max_num_nodes=N)
        adj_local_r = to_dense_adj(edge_index_local, edge_attr=edge_type_r, max_num_nodes=N)
        adj_local_p = to_dense_adj(edge_index_local, edge_attr=edge_type_p, max_num_nodes=N)
    
        adj_global_r = torch.where(adj_local_r!=0, adj_local_r, adj_global)
        adj_global_p = torch.where(adj_local_p!=0, adj_local_p, adj_global)
        edge_index_global_r, edge_type_global_r = dense_to_sparse(adj_global_r)
        edge_index_global_p, edge_type_global_p = dense_to_sparse(adj_global_p)
        edge_type_global_r[edge_type_global_r < 0] = 0
        edge_type_global_p[edge_type_global_p < 0] = 0
        edge_index_global = edge_index_global_r

        edge_length_global = get_distance(pos, edge_index_global).unsqueeze(-1)   # (E, 1)


        # Emb time_step
        #sigma_edge = torch.ones(size=(edge_index_global.size(1), 1), device=pos.device)  # (E, 1)
        sigma_edge = 1.0
        
        edge_attr_r = self.bond_emb(edge_type_r)
        edge_attr_p = self.bond_emb(edge_type_p)
        edge_attr_global_r = self.bond_emb(edge_type_global_r)
        edge_attr_global_p = self.bond_emb(edge_type_global_p)
        #node_attr = self.atom_emb(atom_type)

        atom_emb = self.atom_embedding(atom_type) 
        atom_feat_emb_r = self.atom_feat_embedding(r_feat.float())
        atom_feat_emb_p = self.atom_feat_embedding(p_feat.float())
        z = torch.cat([atom_emb + atom_feat_emb_r, atom_feat_emb_p - atom_feat_emb_r], dim=-1)
        # node_embedding
        node_attr = self.graph_encoder(
                #node_attr,
                z,
                edge_index_local, 
                edge_attr_r, 
                edge_attr_p, 
                edge_index_global,
                pos,
                )
        
        #edge_pair = edge_attr_r * edge_attr_p
        #node_pair = node_attr[edge_index_local[0]] * node_attr[edge_index_local[1]]
        edge_pair = edge_attr_global_r * edge_attr_global_p
        node_pair = node_attr[edge_index_global[0]] * node_attr[edge_index_global[1]]
        h_pair = torch.cat([node_pair, edge_pair], dim=-1)
        ## Assemble pairwise features
        #edge_attr = torch.cat([edge_attr_r, edge_attr_p], dim=-1)
        #h_pair = assemble_atom_pair_feature(
        #    node_attr=node_attr,
        #    edge_index=edge_index_local,
        #    edge_attr=edge_attr,
        #)    # (E_global, 2H)

        ## Invariant features of edges (radius graph, global)
        edge_inv = self.grad_dist_mlp(h_pair) * (1.0 / sigma_edge)    # (E_global, 1)
        
        if return_edges:
            #edge_type_local = torch.stack([edge_type_r, edge_type_p])
            #edge_length = get_distance(pos, edge_index_local).unsqueeze(-1)
            edge_type_global = torch.stack([edge_type_global_r, edge_type_global_p])

            #return edge_inv, edge_index_local, edge_type_local, edge_length
            return edge_inv, edge_index_global, edge_type_global, edge_length_global
        else:
            return edge_inv 
    

    def get_loss(self, atom_type, r_feat, p_feat, pos, bond_index, bond_type, batch, 
                 num_nodes_per_graph, num_graphs, anneal_power=2.0):

        return self.get_loss_diffusion(atom_type, r_feat, p_feat, pos, bond_index, bond_type, 
                                       batch, num_nodes_per_graph, num_graphs, 
                                       anneal_power)

    def get_loss_diffusion(self, atom_type, r_feat, p_feat, pos, bond_index, bond_type, batch, 
                           num_nodes_per_graph, num_graphs, anneal_power=2.0,):

        N = atom_type.size(0)
        node2graph = batch

        # Four elements for DDPM: original_data(pos), gaussian_noise(pos_noise), beta(sigma), time_step
        # Sample noise levels
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs//2+1, ), device=pos.device)
        time_step = torch.cat(
            [time_step, self.num_timesteps-time_step-1], dim=0)[:num_graphs]
        a = self.alphas.index_select(0, time_step)  # (G, )

        # Perterb pos
        a_pos = a.index_select(0, node2graph).unsqueeze(-1)  # (N, 1)
        pos_noise = torch.zeros(size=pos.size(), device=pos.device)
        pos_noise.normal_()
        pos_perturbed = pos + pos_noise * (1.0 - a_pos).sqrt() / a_pos.sqrt()

        # Update invariant edge features, as shown in equation 5-7
        edge_inv, edge_index, edge_type, edge_length  = self(
            atom_type = atom_type,
            r_feat = r_feat,
            p_feat = p_feat,
            pos = pos_perturbed,
            bond_index = bond_index,
            bond_type = bond_type,
            batch = batch,
            time_step = time_step,
            return_edges = True,
        )   # (E_global, 1), (E_local, 1)

        # Compute sigmas_edge
        edge2graph = node2graph.index_select(0, edge_index[0])
        a_edge = a.index_select(0, edge2graph).unsqueeze(-1)  # (E, 1)

        # Compute original and perturbed distances
        d_gt = get_distance(pos, edge_index).unsqueeze(-1)   # (E, 1)
        d_perturbed = edge_length
        d_target = (d_gt - d_perturbed) / (1.0 - a_edge).sqrt() * a_edge.sqrt() # (E_global, 1), denoising direction
        
        #print(d_target.shape)
        #print(d_target.norm(dim=-1).mean())
        #print(edge_inv.norm(dim=-1).mean())

        target_pos_grad = eq_transform(d_target, pos_perturbed, edge_index, edge_length)
        pred_pos_grad = eq_transform(edge_inv, pos_perturbed, edge_index, edge_length)
        #print(target_pos_grad.norm(dim=-1).mean())
        #print(pred_pos_grad.norm(dim=-1).mean())
        #print("--------------------------------")
        loss = (target_pos_grad - pred_pos_grad)**2
        loss = torch.sum(loss, dim=-1, keepdim=True)
        return loss


    def langevin_dynamics_sample(
            self, atom_type, r_feat, p_feat, pos_init, bond_index, bond_type, batch, num_graphs, 
            extend_order, extend_radius=True, n_steps=100, step_lr=1e-6, 
            clip=1000, clip_local=None, clip_pos=None, min_sigma=0, 
            global_start_sigma=float('inf'), w_global=0.2, w_reg=1.0, 
            **kwargs):
        
        return self.langevin_dynamics_sample_diffusion(
                atom_type, r_feat, p_feat, pos_init, bond_index, bond_type, batch, num_graphs, 
                extend_order, extend_radius, n_steps, step_lr, clip, 
                clip_local, clip_pos, min_sigma, global_start_sigma, w_global, 
                w_reg, sampling_type=kwargs.get("sampling_type", 'ddpm_noisy'), 
                eta=kwargs.get("eta", 1.))


    def langevin_dynamics_sample_diffusion(
            self, atom_type, r_feat, p_feat, pos_init, bond_index, bond_type, batch, num_graphs, 
            extend_order, extend_radius=True, n_steps=100, step_lr=1e-6, 
            clip=1000, clip_local=None, clip_pos=None, min_sigma=0, 
            global_start_sigma=float('inf'), w_global=0.2, w_reg=1.0, 
            **kwargs):

        def compute_alpha(beta, t):
            beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
            a = (1 - beta).cumprod(dim=0).index_select(0, t + 1)  # .view(-1, 1, 1, 1)
            return a
        
        sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()
        pos_traj = []
        with torch.no_grad():
            # skip = self.num_timesteps // n_steps
            # seq = range(0, self.num_timesteps, skip)

            ## to test sampling with less intermediate diffusion steps
            # n_steps: the num of steps
            seq = range(self.num_timesteps-n_steps, self.num_timesteps)
            seq_next = [-1] + list(seq[:-1])
            
            pos = pos_init * sigmas[-1]
            for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), desc='sample'):
                t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=pos.device)

                edge_inv, edge_index, edge_type, edge_length  = self(
                    atom_type=atom_type,
                    r_feat = r_feat,
                    p_feat = p_feat,
                    pos=pos,
                    bond_index=bond_index,
                    bond_type=bond_type,
                    batch=batch,
                    time_step=t,
                    return_edges=True,
                    extend_order=extend_order,
                    extend_radius=extend_radius,
                )   # (E_global, 1), (E_local, 1)

                # Local
                node_eq_global = eq_transform(edge_inv, pos, edge_index, edge_length)
                node_eq_global = clip_norm(node_eq_global, limit=clip)
                eps_pos = node_eq_global 
                # Update

                sampling_type = kwargs.get("sampling_type", 'ddpm_noisy')  # types: generalized, ddpm_noisy, ld

                noise = torch.randn_like(pos)  #  center_pos(torch.randn_like(pos), batch)
                if sampling_type in ['generalized','ddpm_noisy','ddpm_det']:
                    b = self.betas
                    t = t[0]
                    next_t = (torch.ones(1) * j).to(pos.device)
                    at = compute_alpha(b, t.long())
                    at_next = compute_alpha(b, next_t.long())
                    if sampling_type == 'generalized':
                        eta = kwargs.get("eta", 1.)
                        et = -eps_pos
                        ## original
                        # pos0_t = (pos - et * (1 - at).sqrt()) / at.sqrt()
                        ## reweighted
                        # pos0_t = pos - et * (1 - at).sqrt() / at.sqrt()
                        c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                        c2 = ((1 - at_next) - c1 ** 2).sqrt()
                        # pos_next = at_next.sqrt() * pos0_t + c1 * noise + c2 * et
                        # pos_next = pos0_t + c1 * noise / at_next.sqrt() + c2 * et / at_next.sqrt()

                        # pos_next = pos + et * (c2 / at_next.sqrt() - (1 - at).sqrt() / at.sqrt()) + noise * c1 / at_next.sqrt()
                        step_size_pos_ld = step_lr * (sigmas[i] / 0.01) ** 2 / sigmas[i]
                        step_size_pos_generalized = 5 * ((1 - at).sqrt() / at.sqrt() - c2 / at_next.sqrt())
                        step_size_pos = step_size_pos_ld if step_size_pos_ld<step_size_pos_generalized else step_size_pos_generalized

                        step_size_noise_ld = torch.sqrt((step_lr * (sigmas[i] / 0.01) ** 2) * 2)
                        step_size_noise_generalized = 3 * (c1 / at_next.sqrt())
                        step_size_noise = step_size_noise_ld if step_size_noise_ld<step_size_noise_generalized else step_size_noise_generalized

                        pos_next = pos - et * step_size_pos +  noise * step_size_noise

                    elif sampling_type == 'ddpm_det':
                        atm1 = at_next
                        beta_t = 1 - at / atm1
                        e = -eps_pos
                        pos0_from_e = (1.0 / at).sqrt() * pos - (1.0 / at - 1).sqrt() * e
                        mean_eps = (
                            (atm1.sqrt() * beta_t) * pos0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * pos
                        ) / (1.0 - at)
                        mean = mean_eps
                        mask = 1 - (t == 0).float()
                        logvar = (beta_t * (1 - atm1) / (1 - at)).log()
                        
                        pos_next = mean + mask * torch.exp(0.5 * logvar) * noise

                    elif sampling_type == 'ddpm_noisy':
                        atm1 = at_next
                        beta_t = 1 - at / atm1
                        e = -eps_pos
                        pos0_from_e = (1.0 / at).sqrt() * pos - (1.0 / at - 1).sqrt() * e
                        mean_eps = (
                            (atm1.sqrt() * beta_t) * pos0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * pos
                        ) / (1.0 - at)
                        mean = mean_eps
                        mask = 1 - (t == 0).float()
                        logvar = beta_t.log()

                        pos_next = mean + mask * torch.exp(0.5 * logvar) * noise

                elif sampling_type == 'ld':
                    step_size = step_lr * (sigmas[i] / 0.01) ** 2
                    pos_next = pos + step_size * eps_pos / sigmas[i] + noise * torch.sqrt(step_size*2)

                pos = pos_next
                if torch.isnan(pos).any():
                    print('NaN detected. Please restart.')
                    raise FloatingPointError()
                pos = center_pos(pos, batch)
                if clip_pos is not None:
                    pos = torch.clamp(pos, min=-clip_pos, max=clip_pos)
                pos_traj.append(pos.clone().cpu())
            
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


def is_train_edge(edge_index, is_sidechain):
    if is_sidechain is None:
        return torch.ones(edge_index.size(1), device=edge_index.device).bool()
    else:
        is_sidechain = is_sidechain.bool()
        return torch.logical_or(is_sidechain[edge_index[0]], is_sidechain[edge_index[1]])


def regularize_bond_length(edge_type, edge_length, rng=5.0):
    mask = is_bond(edge_type).float().reshape(-1, 1)
    d = -torch.clamp(edge_length - rng, min=0.0, max=float('inf')) * mask
    return d


def center_pos(pos, batch):
    pos_center = pos - scatter_mean(pos, batch, dim=0)[batch]
    return pos_center


def clip_norm(vec, limit, p=2):
    norm = torch.norm(vec, dim=-1, p=2, keepdim=True)
    denom = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * denom
