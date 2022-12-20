import torch
import time
import tqdm
from tsdiff.noise_attack.D2X import (
    get_coord_from_distance,
    get_coord_from_distance_parallel,
    get_distance_matrix,
)


def interpolate(
    r_xyz,
    p_xyz,
    alpha,
    ts_xyz=None,
):
    D_r = torch.cdist(r_xyz, r_xyz)
    D_p = torch.cdist(p_xyz, p_xyz)
    D_alpha = (1 - alpha) * D_r + alpha * D_p
    if ts_xyz is not None:
        D_ts = torch.cdist(ts_xyz, ts_xyz)
        gamma = 4 * D_ts - 2 * D_r - 2 * D_p
        D_alpha += alpha * (1 - alpha) * gamma
    return D_alpha


def add_noise_to_distance(
    r_xyz, p_xyz, ts_xyz=None, only_ts=False, parallel=True, **kwargs
):

    if parallel:
        if ts_xyz is not None:
            return _add_noise_to_distance_qst_parallel(
                r_xyz, p_xyz, ts_xyz, only_ts=only_ts, **kwargs
            )
        else:
            return _add_noise_to_distance_lst_parallel(
                r_xyz, p_xyz, only_ts=only_ts, **kwargs
            )
    else:
        if ts_xyz is not None:
            return _add_noise_to_distance_qst(
                r_xyz, p_xyz, ts_xyz, only_ts=only_ts, **kwargs
            )
        else:
            return _add_noise_to_distance_lst(r_xyz, p_xyz, only_ts=only_ts, **kwargs)


def _batching(iterable1, iterable2, batch_size=1000):

    assert len(iterable1) == len(iterable2)
    n = len(iterable1)
    cnt = 0
    while cnt < n:
        if cnt + batch_size <= n:
            yield iterable1[cnt : cnt + batch_size], iterable2[cnt : cnt + batch_size]
            cnt += batch_size
        else:
            yield iterable1[cnt:], iterable2[cnt:]
            cnt += batch_size


def _add_noise_to_distance_qst_parallel(
    r_xyz,
    p_xyz,
    ts_xyz,
    sigma=0.1,
    steps=3000,
    patience=300,
    device="cpu",
    only_ts=False,
    random=True,
    batch_size=1000,
):
    if only_ts:
        x = 0.5
        alpha = x + torch.randn(1).item() * sigma if random else x + sigma
        d_noised = [
            interpolate(r, p, alpha, ts_xyz=ts)
            for r, p, ts in zip(r_xyz, p_xyz, ts_xyz)
        ]
        xyzs = ts_xyz
        out = [[]]
        for batch_idx, (xyz_batch, d_noised_batch) in tqdm.tqdm(
            enumerate(_batching(xyzs, d_noised, batch_size=batch_size))
        ):
            new_xyz = get_coord_from_distance_parallel(
                xyz_batch, d_noised_batch, steps=steps, patience=patience, device=device
            )
            out[0].extend(new_xyz)
    else:
        out = [[], []]
        x = 0
        alpha = x + torch.randn(1).item() * sigma if random else x + sigma
        d_noised = [
            interpolate(r, p, alpha, ts_xyz=ts)
            for r, p, ts in zip(r_xyz, p_xyz, ts_xyz)
        ]
        xyzs = r_xyz
        for batch_idx, (xyz_batch, d_noised_batch) in tqdm.tqdm(
            enumerate(_batching(xyzs, d_noised, batch_size=batch_size))
        ):
            new_xyz = get_coord_from_distance_parallel(
                xyz_batch, d_noised_batch, steps=steps, patience=patience, device=device
            )
            out[0].extend(new_xyz)

        x = 1
        alpha = x + torch.randn(1).item() * sigma if random else x + sigma
        d_noised = [
            interpolate(r, p, alpha, ts_xyz=ts)
            for r, p, ts in zip(r_xyz, p_xyz, ts_xyz)
        ]
        xyzs = p_xyz
        for batch_idx, (xyz_batch, d_noised_batch) in tqdm.tqdm(
            enumerate(_batching(xyzs, d_noised, batch_size=batch_size))
        ):
            new_xyz = get_coord_from_distance_parallel(
                xyz_batch, d_noised_batch, steps=steps, patience=patience, device=device
            )
            out[1].extend(new_xyz)

    return out


def _add_noise_to_distance_lst_parallel(
    r_xyz,
    p_xyz,
    sigma=0.1,
    steps=3000,
    patience=300,
    device="cpu",
    only_ts=False,
    random=True,
    batch_size=1000,
):
    if only_ts:
        out = [[]]
        x = 0.5
        alpha = x + torch.randn(1).item() * sigma if random else x + sigma
        xyzs = [(r + p) / 2 for r, p in zip(r_xyz, p_xyz)]
        d_noised = [interpolate(r, p, alpha) for r, p in zip(r_xyz, p_xyz)]
        out = [[]]
        for batch_idx, (xyz_batch, d_noised_batch) in tqdm.tqdm(
            enumerate(_batching(xyzs, d_noised, batch_size=batch_size))
        ):
            new_xyz = get_coord_from_distance_parallel(
                xyz_batch, d_noised_batch, steps=steps, patience=patience, device=device
            )
            out[0].extend(new_xyz)

    else:
        out = [[], []]
        x = 0
        alpha = x + torch.randn(1).item() * sigma if random else x + sigma
        d_noised = [interpolate(r, p, alpha) for r, p in zip(r_xyz, p_xyz)]
        xyzs = r_xyz
        for batch_idx, (xyz_batch, d_noised_batch) in tqdm.tqdm(
            enumerate(_batching(xyzs, d_noised, batch_size=batch_size))
        ):
            new_xyz = get_coord_from_distance_parallel(
                xyz_batch, d_noised_batch, steps=steps, patience=patience, device=device
            )
            out[0].extend(new_xyz)

        x = 1
        alpha = x + torch.randn(1).item() * sigma if random else x + sigma
        d_noised = [interpolate(r, p, alpha) for r, p in zip(r_xyz, p_xyz)]
        xyzs = p_xyz
        for batch_idx, (xyz_batch, d_noised_batch) in tqdm.tqdm(
            enumerate(_batching(xyzs, d_noised, batch_size=batch_size))
        ):
            new_xyz = get_coord_from_distance_parallel(
                xyz_batch, d_noised_batch, steps=steps, patience=patience, device=device
            )
            out[1].extend(new_xyz)

    return out


def _add_noise_to_distance_qst(
    r_xyz,
    p_xyz,
    ts_xyz,
    sigma=0.1,
    steps=3000,
    patience=300,
    device="cpu",
    only_ts=False,
    random=True,
):
    if only_ts:
        xyzs, sigmas = [ts_xyz], [0.5]
    else:
        xyzs, sigmas = [r_xyz, p_xyz], [0, 1]

    out = []
    for xyz, x in zip(xyzs, sigmas):
        if random:
            alpha = torch.randn(1).item() * sigma + x
        else:
            alpha = x + sigma
        D_noise = interpolate(r_xyz, p_xyz, alpha, ts_xyz=ts_xyz)
        xyz_ = get_coord_from_distance(
            xyz, D_noise, steps=steps, patience=patience, device=device
        )
        out.append(xyz_)

    return out


def _add_noise_to_distance_lst(
    r_xyz,
    p_xyz,
    sigma=0.1,
    steps=3000,
    patience=300,
    device="cpu",
    only_ts=False,
    random=True,
):
    if only_ts:
        xyzs, sigmas = [(r_xyz + p_xyz) / 2], [0.5]
    else:
        xyzs, sigmas = [r_xyz, p_xyz], [0, 1]

    out = []
    for xyz, x in zip(xyzs, sigmas):
        if random:
            alpha = torch.randn(1).item() * sigma + x
        else:
            alpha = x + sigma
        alpha = torch.randn(1).item() * sigma + x
        D_noise = interpolate(r_xyz, p_xyz, alpha)
        xyz_ = get_coord_from_distance(
            xyz, D_noise, steps=steps, patience=patience, device=device
        )
        out.append(xyz_)

    return out


def xyz_block_to_pos(block, dtype=torch.float):
    lines = block.strip().split("\n")
    xyz = []
    for l in lines[2:]:
        a_xyz = [float(v) for v in l.strip().split("\t")[1:]]
        xyz.append(a_xyz)
    return torch.DoubleTensor(xyz).to(dtype)


def read_xyz_file(xyz_file):
    with open(xyz_file, "r") as f:
        lines = f.readlines()
    xyz_blocks = []
    N = 0
    for l in lines:
        try:
            n = int(l.strip()) + 2
            xyz_blocks.append("".join(lines[N : N + n]))
            N += n
        except:
            pass
    return xyz_blocks


def reform_xyz_block(xyz_block, new_pos: torch.Tensor):
    lines = xyz_block.strip().split("\n")
    new_lines = lines[:2]

    assert len(new_pos) == len(lines[2:])
    for l, a_pos in zip(lines[2:], new_pos):
        a_pos = "\t".join([str(i.item()) for i in a_pos])
        a_line = "\t".join([l.split("\t")[0], a_pos])
        new_lines.append(a_line)
    new_xyz_block = "\n".join(new_lines)
    return new_xyz_block


if __name__ == "__main__":
    import argparse
    import multiprocessing as mp
    import os

    str2bool = lambda x: True if x == "true" or x == "True" else False
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", "-sigma", required=True, type=float)
    parser.add_argument("--random", "-random", default=True, type=str2bool)
    parser.add_argument(
        "--interpolation", "-interpolation", default="qst", choices=["qst", "lst"]
    )
    parser.add_argument("--device", "-device", default="cpu", type=str)
    parser.add_argument(
        "--r_path",
        "-r_path",
        default="/home/ksh/MolDiff/tsdiff/"
        "data/TS/b97d3/random_split/"
        "raw_data/b97d3_r_test.xyz",
        type=str,
    )
    parser.add_argument(
        "--p_path",
        "-p_path",
        default="/home/ksh/MolDiff/tsdiff"
        "/data/TS/b97d3/random_split/"
        "raw_data/b97d3_p_test.xyz",
        type=str,
    )
    parser.add_argument(
        "--ts_path",
        "-ts_path",
        default="/home/ksh/MolDiff/tsdiff/"
        "data/TS/b97d3/random_split/"
        "raw_data/b97d3_ts_test.xyz",
        type=str,
    )
    parser.add_argument(
        "--save_type", "-save_type", default="pkl", choices=["pkl", "xyz"]
    )
    parser.add_argument(
        "--save_prefix", "-save_prefix", default="b97d3_random_split", type=str
    )

    parser.add_argument(
        "--feat_dict",
        "-feat_dict",
        default="../data/TS/b97d3/random_split/feat_dict.pkl",
        type=str,
    )
    parser.add_argument("--seed", "-seed", type=int, default=0)
    parser.add_argument("--batch_size", "-batch_size", type=int, default=100)
    parser.add_argument("--n", "-n", default=1, type=int, help="Deprecated")
    parser.add_argument("--steps", "-steps", default=3000, type=int)
    parser.add_argument("--patience", "-patience", default=300, type=int)
    args = parser.parse_args()

    # reproducibility
    torch.manual_seed(args.seed)
    st = time.time()

    # read reactant & product test dataset to add noise (.xyz file)
    r_blocks = read_xyz_file(args.r_path)
    p_blocks = read_xyz_file(args.p_path)
    ts_blocks = read_xyz_file(args.ts_path)

    # extract position array from .xyz file (torch Tensor)
    r_pos_list = [xyz_block_to_pos(b, dtype=torch.double) for b in r_blocks]
    p_pos_list = [xyz_block_to_pos(b, dtype=torch.double) for b in p_blocks]
    if args.interpolation == "qst":
        ts_pos_list = [xyz_block_to_pos(b, dypte=torch.double) for b in ts_blocks]
    else:
        ts_pos_list = None

    # add noise
    ret = add_noise_to_distance(
        r_pos_list,
        p_pos_list,
        ts_xyz=ts_pos_list,
        only_ts=args.save_type == "pkl",
        random=args.random,
        sigma=args.sigma,
        steps=args.steps,
        patience=args.patience,
        device=args.device,
    )

    if args.save_type == "pkl":
        # only retuns noised TS structure.
        noised_pos_list = ret[0]
    else:
        # only retuns noised R,P structure.
        r_data_iter, p_data_iter = ret
        noised_pos_list = zip(r_data_iter, p_data_iter)
    
    # save path setting
    print(f"Done parallel optimization {(time.time() - st)/60:0.2f}")
    save_dir = (
        f"{args.save_prefix}_{args.interpolation}/"
        f"{args.save_type}_data/"
        f"{'' if args.random else 'fixed_'}sigma_{args.sigma:0.1e}"
    )
    _ = save_dir.split("/")
    _ = ["/".join(_[:i]) for i in range(1, len(_) + 1)]
    for p in _:
        if not os.path.isdir(p):
            os.system(f"mkdir {p}")

    # save
    if args.save_type == "pkl":
        import pickle
        from tsdiff.utils.datasets import generate_ts_data2

        # load feature dict
        with open(args.feat_dict, "rb") as f:
            feat_dict = pickle.load(f)
        data_list = []
        # extract smiles of r, p. required for generating 2D-graph data.
        r_smarts = [b.split("\n")[1].strip() for b in r_blocks]
        p_smarts = [b.split("\n")[1].strip() for b in p_blocks]

        # generate data (atom feature is not yet one-hot)
        for r_smi, p_smi, r_pos, p_pos, xyz_gt, noised_pos in zip(
            r_smarts, p_smarts, r_pos_list, p_pos_list, ts_blocks, noised_pos_list
        ):
            data, _ = generate_ts_data2(r_smi, p_smi, xyz_gt, feat_dict=feat_dict)
            data.ts_guess = noised_pos.float()
            data.pos_r = r_pos.float()
            data.pos_p = p_pos.float()
            data_list.append(data)

        # atom feature conversion to one-hot
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

        # save
        with open(f"{save_dir}/data_{args.seed}.pkl", "wb") as f:
            pickle.dump(data_list, f)

    elif args.save_type == "xyz":
        r_msg, p_msg = "", ""
        for r_xyz, p_xyz, (noised_r_pos, noised_p_pos) in zip(
            r_blocks, p_blocks, noised_pos_list
        ):
            noised_r_xyz = reform_xyz_block(r_xyz, noised_r_pos)
            noised_p_xyz = reform_xyz_block(p_xyz, noised_p_pos)
            r_msg += noised_r_xyz + "\n$$$$$\n"
            p_msg += noised_p_xyz + "\n$$$$$\n"

            with open(f"{save_dir}/r_seed_{args.seed}.xyz", "w") as f:
                f.write(r_msg.strip())
            with open(f"{save_dir}/p_seed_{args.seed}.xyz", "w") as f:
                f.write(p_msg.strip())
