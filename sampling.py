import os
import argparse
import pickle
import torch
from tqdm.auto import tqdm
from easydict import EasyDict

from models import sampler
from torch_geometric.transforms import Compose
from models.epsnet import get_model
from utils.datasets import generate_ts_data2
from utils.transforms import CountNodesPerGraph
from utils.misc import seed_all, get_logger
from torch_geometric.data import Batch


def num_confs(num: str):
    if num.endswith("x"):
        return lambda x: x * int(num[:-1])
    elif int(num) > 0:
        return lambda x: int(num)
    else:
        raise ValueError()


def repeat(iterable, num):
    new_iterable = []
    for x in iterable:
        new_iterable.extend([x.clone() for _ in range(num)])
    return new_iterable


def batching(iterable, batch_size, repeat_num=1):
    cnt = 0
    iterable = repeat(iterable, repeat_num)
    while cnt < len(iterable):
        if cnt + batch_size <= len(iterable):
            yield iterable[cnt: cnt + batch_size]
            cnt += batch_size
        else:
            yield iterable[cnt:]
            cnt += batch_size


def preprocessing(smarts_list, feat_dict_path="feat_dict.pkl"):
    feat_dict = pickle.load(open(feat_dict_path, "rb"))

    data_list = []
    for a_smarts in smarts_list:
        r, p = a_smarts.split(">>")
        data, _ = generate_ts_data2(r, p, None, feat_dict=feat_dict)
        data_list.append(data)
    num_cls = [len(v) for k, v in feat_dict.items()]
    for data in data_list:
        feat_onehot = []
        feats = data.r_feat.T
        for feat, n_cls in zip(feats, num_cls):
            feat_onehot.append(torch.nn.functional.one_hot(feat, num_classes=n_cls))
        data.r_feat = torch.cat(feat_onehot, dim=-1)

        feat_onehot = []
        feats = data.p_feat.T
        for feat, n_cls in zip(feats, num_cls):
            feat_onehot.append(torch.nn.functional.one_hot(feat, num_classes=n_cls))
        data.p_feat = torch.cat(feat_onehot, dim=-1)

    return data_list


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
    # Model
    parser.add_argument("ckpt", type=str, help="path for loading the checkpoint", nargs="+")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--resume", action="store_true", default=None)

    # IO parameters
    parser.add_argument("--save_traj", action="store_true", default=False, help="whether store the whole trajectory for sampling",)
    parser.add_argument("--save_dir", type=str, required=True)

    # Test data parameters
    parser.add_argument("--feat_dict", "-feat_dict", type=str, default="./data/TS/wb97xd3/random_split_42/feat_dict.pkl")
    parser.add_argument("--test_set", type=str, required=True, help="1. file of smarts string (.txt) 2. preprocessed file (.pck or .pkl) 3. a smarts string (with out file format)")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=9999)
    parser.add_argument("--repeat", type=int, default=1)

    # Guess TS parameters
    parser.add_argument("--from_ts_guess", action="store_true", default=False)
    parser.add_argument("--denoise_from_time_t", type=int, default=None)
    parser.add_argument("--noise_from_time_t", type=int, default=None)

    # Sampling parameters
    parser.add_argument("--clip", type=float, default=1000.0)
    parser.add_argument("--n_steps", type=int, default=5000, help="sampling num steps; for DSM framework, this means num steps for each noise scale",)

    # Parameters for DDPM
    parser.add_argument("--sampling_type", type=str, default="ld", help="generalized, ddpm_noisy, ddpm_det, ld: sampling method for DDIM, DDPM or Langevin Dynamics",)
    parser.add_argument("--eta", type=float, default=1.0, help="weight for DDIM and DDPM: 0->DDIM, 1->DDPM",)
    parser.add_argument("--step_lr", type=float, default=1e-7, help="step_lr of sampling",)

    parser.add_argument("--seed", type=int, default=2022, help="seed number for random",)
    args = parser.parse_args()

    # Logging
    # log_dir = args.save_dir
    log_dir = args.save_dir
    os.system(f"mkdir -p {log_dir}")
    logger = get_logger("test", log_dir)
    logger.info(args)

    # Load checkpoint
    logger.info("Loading model...")
    ckpts = [torch.load(x) for x in args.ckpt]
    models = []
    for ckpt, ckpt_path in zip(ckpts, args.ckpt):
        logger.info(f"load model from {ckpt_path}")
        model = get_model(ckpt["config"].model).to(args.device)
        model.load_state_dict(ckpt["model"])
        models.append(model)

    model = sampler.EnsembleSampler(models).to(args.device)
    seed_all(args.seed)

    # Datasets and loaders
    logger.info("Loading datasets...")
    transforms = Compose([CountNodesPerGraph(), ])

    if ".txt" in args.test_set or ".pck" in args.test_set or ".pkl" in args.test_set:
        if not os.path.isfile(args.test_set):
            logger.info(f"!!!Test file {args.test_set} is not found!!!\n" * 3)
            exit()
        elif ".txt" in args.test_set:
            logger.info(f"Test file from {args.test_set}.\n Processing smarts...")
            smarts_list = open(args.test_set, "r").read().strip().split("\n")
            test_set = preprocessing(smarts_list, feat_dict_path=args.feat_dict)
        else:
            logger.info(f"Test file from {args.test_set}.\n Loading dataset...")
            test_set = pickle.load(open(args.test_set, "w"))
    else:
        logger.info(f"Test smarts : {args.test_set}.\n Processing smarts...")
        smarts_list = [args.test_set]
        test_set = preprocessing(smarts_list, feat_dict_path=args.feat_dict)

    test_set_selected = []
    for i, data in enumerate(test_set):
        if not (args.start_idx <= i < args.end_idx):
            continue
        test_set_selected.append(data)

    done_smiles = set()
    results = []
    if args.resume is not None:
        with open(args.resume, "rb") as f:
            results = pickle.load(f)
        for data in results:
            done_smiles.add(data.smiles)

    for i, batch in tqdm(enumerate(batching(test_set_selected, args.batch_size, repeat_num=args.repeat))):
        batch = Batch.from_data_list(batch).to(args.device)
        for _ in range(2):  # Maximum number of retry
            try:
                if args.from_ts_guess:
                    # print("Geometry Generation with Guess TS Support")
                    assert args.denoise_from_time_t is not None
                    if hasattr(batch, "ts_guess"):
                        init_guess = batch.ts_guess
                    else:
                        init_guess = batch.pos
                    start_t = (
                        args.noise_from_time_t
                        if args.noise_from_time_t is not None
                        else args.denoise_from_time_t
                    )
                    sqrt_a = model.alphas[start_t - 1].sqrt() if start_t != 0 else 1
                    init_guess = init_guess / sqrt_a
                    pos_init = init_guess.to(args.device)

                else:
                    pos_init = torch.randn(batch.num_nodes, 3).to(args.device)

                pos_gen, pos_gen_traj = model.dynamic_sampling(
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
                    clip=args.clip,
                    sampling_type=args.sampling_type,
                    eta=args.eta,
                    noise_from_time_t=args.noise_from_time_t,
                    denoise_from_time_t=args.denoise_from_time_t,
                )
                alphas = model.alphas.detach()
                if args.denoise_from_time_t is not None:
                    alphas = alphas[args.denoise_from_time_t - args.n_steps: args.denoise_from_time_t]
                else:
                    alphas = alphas[model.num_timesteps - args.n_steps: model.num_timesteps]
                alphas = alphas.flip(0).view(-1, 1, 1)
                pos_gen_traj_ = torch.stack(pos_gen_traj) * alphas.sqrt().cpu()

                for j, data in enumerate(batch.to_data_list()):
                    mask = batch.batch == j
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
                clip = 20
                logger.warning("Retrying with clipping thresh 20.")

    os.system(f"rm {save_path}")
    save_path = os.path.join(log_dir, "samples_all.pkl")
    logger.info("Saving samples to: %s" % save_path)

    with open(save_path, "wb") as f:
        pickle.dump(results, f)
