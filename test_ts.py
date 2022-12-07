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
from utils.transforms import CountNodesPerGraph, AddHigherOrderEdges
from utils.misc import seed_all, get_new_log_dir, seed_all, get_logger, repeat_data


def num_confs(num: str):
    if num.endswith("x"):
        return lambda x: x * int(num[:-1])
    elif int(num) > 0:
        return lambda x: int(num)
    else:
        raise ValueError()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str, help="path for loading the checkpoint")
    parser.add_argument("--save_traj", action="store_true", default=False, help="whether store the whole trajectory for sampling")
    parser.add_argument("--save_suffix", type=str, required=True, default="")
    parser.add_argument("--from_ts_guess", action="store_true", default=False)
    parser.add_argument("--from_time_t", type=int, default=None)

    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--num_confs", type=num_confs, default=num_confs("2x"))
    parser.add_argument("--test_set", type=str, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--clip", type=float, default=1000.0)
    parser.add_argument("--n_steps", type=int, default=5000,
        help="sampling num steps; for DSM framework, this means num steps for each noise scale",
    )
    parser.add_argument("--global_start_sigma", type=float, default=0.5,
        help="enable global gradients only when noise is low",
    )
    parser.add_argument("--w_global", type=float, default=1.0, 
            help="weight for global gradients"
    )
    # Parameters for DDPM
    parser.add_argument("--sampling_type", type=str, default="ld",
        help="generalized, ddpm_noisy, ddpm_det, ld: sampling method for DDIM, DDPM or Langevin Dynamics",
    )
    parser.add_argument("--eta", type=float, default=1.0,
        help="weight for DDIM and DDPM: 0->DDIM, 1->DDPM",
    )
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.ckpt)
    config_path = glob(
        os.path.join(os.path.dirname(os.path.dirname(args.ckpt)), "*.yml")
    )[0]
    with open(config_path, "r") as f:
        config = EasyDict(yaml.safe_load(f))
    seed_all(config.train.seed)
    log_dir = os.path.dirname(os.path.dirname(args.ckpt))

    # Logging
    fn = f"{args.sampling_type}_{args.save_suffix}"
    output_dir = get_new_log_dir(root=log_dir, prefix="sample", tag=args.tag, fn=fn)
    #output_dir = None
    logger = get_logger("test", output_dir)
    logger.info(args)

    # Datasets and loaders
    logger.info("Loading datasets...")
    transforms = Compose(
        [
            CountNodesPerGraph(),
            #AddHigherOrderEdges(
            #    order=config.model.edge_order
            #),  # Offline edge augmentation
        ]
    )
    if args.test_set is None:
        # test_set = PackedConformationDataset(config.dataset.test, transform=transforms)
        test_set = TSDataset(config.dataset.test, transform=transforms)
    else:
        # test_set = PackedConformationDataset(args.test_set, transform=transforms)
        test_set = TSDataset(args.test_set, transform=transforms)

    # Model
    logger.info("Loading model...")

    model = get_model(ckpt["config"].model).to(args.device)
    model.load_state_dict(ckpt["model"])

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

    for i, data in enumerate(tqdm(test_set_selected)):
        if data.smiles in done_smiles:
            logger.info("Molecule#%d is already done." % i)
            continue

        # num_refs = data.pos_ref.size(0) // data.num_nodes
        # num_samples = args.num_confs(num_refs)
        num_samples = args.num_confs(1)

        data_input = data.clone()
        batch = repeat_data(data_input, num_samples).to(args.device)

        clip_local = None
        for _ in range(2):  # Maximum number of retry
            try:
                if args.from_ts_guess:
                    #print("Geometry Generation with Guess TS Support")
                    assert args.from_time_t is not None
                    if hasattr(batch, "ts_guess"):
                        init_guess = batch.ts_guess
                    else:
                        #print("No TS guess is given, start with ground truth TS Support")
                        init_guess = batch.pos
                    #print(init_guess)
                    #print(init_guess.shape)
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
                    step_lr=1e-6,
                    w_global=args.w_global,
                    global_start_sigma=args.global_start_sigma,
                    clip=args.clip,
                    clip_local=clip_local,
                    sampling_type=args.sampling_type,
                    eta=args.eta,
                    from_time_t=args.from_time_t,
                )
                pos_gen = pos_gen.cpu()
                if args.save_traj:
                    data.pos_gen = torch.stack(pos_gen_traj)
                else:
                    data.pos_gen = pos_gen
                results.append(data)
                done_smiles.add(data.smiles)

                save_path = os.path.join(output_dir, "samples_%d.pkl" % i)
                logger.info("Saving samples to: %s" % save_path)
                with open(save_path, "wb") as f:
                    pickle.dump(results, f)

                break  # No errors occured, break the retry loop
            except FloatingPointError:
                clip_local = 20
                logger.warning("Retrying with local clipping.")
    
    #with open("test_save.pkl", "wb") as f:
    #    pickle.dump(results, f)
    save_path = os.path.join(output_dir, "samples_all.pkl")
    logger.info("Saving samples to: %s" % save_path)

    def get_mol_key(data):
        for i, d in enumerate(test_set_selected):
            if d.smiles == data.smiles:
                return i
        return -1

    results.sort(key=get_mol_key)

    with open(save_path, "wb") as f:
        pickle.dump(results, f)
