import os
import argparse
import pickle
import yaml
import torch
from glob import glob
from tqdm.auto import tqdm
from easydict import EasyDict

from torch_geometric.transforms import Compose
from models.epsnet import get_model
from utils.datasets import TSDataset
from utils.transforms import CountNodesPerGraph
from utils.misc import seed_all, get_logger
from torch_geometric.data import Batch

torch.Tensor(0).to("cuda")


def num_confs(num: str):
    if num.endswith("x"):
        return lambda x: x * int(num[:-1])
    elif int(num) > 0:
        return lambda x: int(num)
    else:
        raise ValueError()


def batching(iterable, batch_size):
    cnt = 0
    while cnt < len(iterable):
        if cnt + batch_size <= len(iterable):
            yield iterable[cnt: cnt + batch_size]
            cnt += batch_size
        else:
            yield iterable[cnt:]
            cnt += batch_size


if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--ckpt", type=str, 
                        help="path for loading the checkpoint", nargs="+")
    parser.add_argument("--save_traj", action="store_true", default=False,
                        help="whether store the whole trajectory for sampling")

    # Parameters for calc utility
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--resume", action="store_true", default=False,)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sampling_even_index", type=str2bool, default=False,)

    # Sampling dataset
    parser.add_argument("--test_set", type=str, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=9999)

    # Save dir
    parser.add_argument("--save_dir", type=str, default=None)

    # Sampling parameters
    parser.add_argument("--clip", type=float, default=1000.0)
    parser.add_argument("--n_steps", type=int, default=5000,
                        help="sampling num steps; for DSM framework, this means num steps for each noise scale")
    parser.add_argument("--global_start_sigma", type=float, default=0.5,
                        help="enable global gradients only when noise is low")
    parser.add_argument("--w_global", type=float, default=1.0,
                        help="weight for global gradients")

    # Parameters for pseudo-TS support and noise
    parser.add_argument("--from_ts_guess", type=str, default="")
    parser.add_argument("--denoise_from_time_t", type=int, default=None)
    parser.add_argument("--noise_from_time_t", type=int, default=None)

    # Parameters for sampling (This could change the sampling results)
    parser.add_argument("--sampling_type", type=str, default="ld",
                        help="generalized, ddpm_noisy, ddpm_det, ld: sampling method for DDIM, DDPM or Langevin Dynamics",)
    parser.add_argument("--eta", type=float, default=1.0,
                        help="weight for DDIM and DDPM: 0->DDIM, 1->DDPM",)
    parser.add_argument("--seed", type=int, default=2022,
                        help="seed number for random",)
    parser.add_argument("--step_lr", type=float, default=1e-6,
                        help="step_lr of sampling",)

    args = parser.parse_args()

    # argument from config file
    if args.config is not None:
        with open(args.config, "r") as f:
            arguments = yaml.safe_load(f)
        for arg, value in arguments.items():
            args.__setattr__(arg, value)

    # Logging
    assert args.save_dir is not None
    log_dir = args.save_dir
    os.system(f"mkdir -p {log_dir}")
    logger = get_logger("test", log_dir)
    logger.info(args)

    # save arguments config
    with open(os.path.join(args.save_dir,"run_config.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    # Load checkpoint and Model
    logger.info("Loading model...")
    ckpts = [torch.load(x) for x in args.ckpt]
    models = []
    for ckpt, ckpt_path in zip(ckpts, args.ckpt):
        logger.info(f"load model from {ckpt_path}")
        config_path = glob(
            os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), "*.yml")
        )[0]
        model = get_model(ckpt["config"].model).to(args.device)
        model.load_state_dict(ckpt["model"])
        models.append(model)

    from models.epsnet import ensemble
    model = ensemble.EnsembleNetwork(models).to(args.device)

    with open(config_path, "r") as f:
        config = EasyDict(yaml.safe_load(f))
    seed_all(args.seed)

    # Datasets and loaders
    logger.info("Loading datasets...")
    transforms = Compose([CountNodesPerGraph(), ])
    if args.test_set is None:
        raise ValueError("Please specify the test set")
    else:
        test_set = TSDataset(args.test_set, transform=transforms)

    done_smiles = set()
    results = []
    if args.resume:
        save_path = os.path.join(log_dir, "samples_not_all.pkl")
        with open(save_path, "rb") as f:
            results = pickle.load(f)

        for data in results:
            done_smiles.add(data.smiles)

    test_set_selected = []
    for i, data in enumerate(test_set):
        if not (args.start_idx <= i < args.end_idx):
            continue
        if args.sampling_even_index:
            if i % 2 != 0:
                continue
        if data.smiles in done_smiles:
            continue
        test_set_selected.append(data)


    for i, batch in tqdm(enumerate(batching(test_set_selected, args.batch_size))):
        batch = Batch.from_data_list(batch).to(args.device)

        clip_local = None
        for _ in range(2):  # Maximum number of retry
            try:
                if args.from_ts_guess:
                    # print("Geometry Generation with Guess TS Support")
                    assert args.denoise_from_time_t is not None
                    if hasattr(batch, args.from_ts_guess):
                        init_guess = batch[args.from_ts_guess]
                    else:
                        raise ValueError(f"no {args.from_ts_guess} in batch")

                    # set parameters
                    start_t = args.denoise_from_time_t
                    if args.noise_from_time_t is not None:
                        start_t = args.noise_from_time_t

                    if start_t != 0:
                        sqrt_a = model.alphas[start_t - 1].sqrt()
                    else:
                        sqrt_a = 1
                    init_guess = init_guess / sqrt_a
                    pos_init = init_guess.to(args.device)
                else:
                    pos_init = torch.randn(batch.num_nodes, 3).to(args.device)

                pos_gen, pos_gen_traj = model.langevin_dynamics_sample(
                    atom_type=batch.atom_type,
                    r_feat=batch.r_feat,
                    p_feat=batch.p_feat,
                    pos_init=pos_init,
                    bond_index=batch.edge_index,
                    bond_type=batch.edge_type,
                    batch=batch.batch,
                    num_graphs=batch.num_graphs,
                    extend_order=True,  # Done in transforms.
                    n_steps=args.n_steps,
                    step_lr=args.step_lr,
                    w_global=args.w_global,
                    global_start_sigma=args.global_start_sigma,
                    clip=args.clip,
                    clip_local=clip_local,
                    sampling_type=args.sampling_type,
                    eta=args.eta,
                    noise_from_time_t=args.noise_from_time_t,
                    denoise_from_time_t=args.denoise_from_time_t,
                )
                alphas = model.alphas.detach()
                if args.denoise_from_time_t is not None:
                    alphas = alphas[
                        args.denoise_from_time_t - args.n_steps: args.denoise_from_time_t
                    ]
                else:
                    alphas = alphas[model.num_timesteps - args.n_steps: model.num_timesteps]
                alphas = alphas.flip(0).view(-1, 1, 1)
                pos_gen_traj_ = torch.stack(pos_gen_traj) * alphas.sqrt().cpu()

                for i, data in enumerate(batch.to_data_list()):
                    mask = batch.batch == i
                    if args.save_traj:
                        data.pos_gen = pos_gen_traj_[:, mask]
                    else:
                        data.pos_gen = pos_gen[mask]

                    data = data.to("cpu")
                    results.append(data)
                    done_smiles.add(data.smiles)

                save_path = os.path.join(log_dir, "samples_not_all.pkl")
                with open(save_path, "wb") as f:
                    pickle.dump(results, f)

                break  # No errors occured, break the retry loop
            except FloatingPointError:
                clip_local = 20
                logger.warning("Retrying with local clipping.")

    os.system(f"rm {save_path}")
    save_path = os.path.join(log_dir, "samples_all.pkl")
    logger.info("Saving samples to: %s" % save_path)

    with open(save_path, "wb") as f:
        pickle.dump(results, f)
