import torch
from GeoDiff.noise_attack.D2X import get_distance_matrix, get_coord_from_ditance


def interpolate(r_xyz, p_xyz, alpha, ts_xyz=None, ):
    D_r = get_distance_matrix(r_xyz)
    D_p = get_distance_matrix(p_xyz)
    D_alpha = (1-alpha) * D_r + alpha * D_p
    if ts_xyz is not None:
        D_ts = get_distance_matrix(ts_xyz)
        gamma = 4 * D_ts - 2 * D_r - 2 * D_p
        D_alpha += alpha*(1-alpha)*gamma
    return D_alpha

def add_noise_to_distance_qst(r_xyz, p_xyz, ts_xyz, alpha, sigma=0.1, steps=3000, patience=300, device="cpu"):
    D_r_noise = interpolate(r_xyz, p_xyz, alpha, ts_xyz=ts_xyz)
    D_p_noise = interpolate(r_xyz, p_xyz, 1+alpha, ts_xyz=ts_xyz)
    D_ts_noise = interpolate(r_xyz, p_xyz, 0.5+alpha, ts_xyz=ts_xyz)
    
    r_xyz_ = get_coord_from_ditance(r_xyz, D_r_noise, steps=steps, patience=patience, device=device)
    p_xyz_ = get_coord_from_ditance(p_xyz, D_p_noise, steps=steps, patience=patience, device=device)
    ts_xyz_ = get_coord_from_ditance(ts_xyz, D_ts_noise, steps=steps, patience=patience, device=device)
    return r_xyz_, p_xyz_, ts_xyz_

def add_noise_to_distance_lst(r_xyz, p_xyz, alpha, sigma=0.1, steps=3000, patience=300, device="cpu"):
    D_r = get_distance_matrix(r_xyz)
    D_p = get_distance_matrix(p_xyz)

    D_r_noise = interpolate(r_xyz, p_xyz, alpha)
    D_p_noise = interpolate(r_xyz, p_xyz, 1+alpha)
    
    r_xyz_ = get_coord_from_ditance(r_xyz, D_r_noise, steps=steps, patience=patience, device=device)
    p_xyz_ = get_coord_from_ditance(p_xyz, D_p_noise, steps=steps, patience=patience, device=device)
    return r_xyz_, p_xyz_

def xyz_block_to_pos(block):
    lines = block.strip().split("\n")
    xyz = []
    for l in lines[2:]:
        a_xyz = [float(v) for v in l.strip().split("\t")[1:]]
        xyz.append(a_xyz)
    return torch.DoubleTensor(xyz)

def read_xyz_file(xyz_file):
    with open(xyz_file,"r") as f:
        lines = f.readlines()
    xyz_blocks = []
    N = 0
    for l in lines:
        try:
            n = int(l.strip())+2
            xyz_blocks.append("".join(lines[N:N+n]))
            N += n
        except:
            pass
    return xyz_blocks

def reform_xyz_block(xyz_block, new_pos :torch.Tensor):
    lines = xyz_block.strip().split("\n")
    new_lines = lines[:2]

    assert len(new_pos) == len(lines[2:])
    for l, a_pos in zip(lines[2:], new_pos):
        a_pos = "\t".join([str(i.item()) for i in a_pos])
        a_line = "\t".join([l.split("\t")[0], a_pos])
        new_lines.append(a_line)
    new_xyz_block = "\n".join(new_lines)
    return new_xyz_block

 

if __name__=="__main__":
    import argparse
    import multiprocessing as mp
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma","-sigma",required=True,type=float)
    parser.add_argument("--n","-n",default=5,type=int)
    parser.add_argument("--steps","-steps",default=3000,type=int)
    parser.add_argument("--patience","-patience",default=300,type=int)
    
    args = parser.parse_args()

    n, sigma, steps, patience = args.n, args.sigma, args.steps, args.patience

    # read reactant & product test dataset to add noise (xyz file)
    r_path = "/home/ksh/MolDiff/GeoDiff/data/TS/b97d3/random_split/raw_data/b97d3_r_test.xyz"
    p_path = "/home/ksh/MolDiff/GeoDiff/data/TS/b97d3/random_split/raw_data/b97d3_p_test.xyz"
    ts_path = "/home/ksh/MolDiff/GeoDiff/data/TS/b97d3/random_split/raw_data/b97d3_ts_test.xyz"
    r_blocks = read_xyz_file(r_path)
    p_blocks = read_xyz_file(p_path)
    ts_blocks = read_xyz_file(ts_path)

    device = "cpu"
    def func(x):
        r_block, p_block, ts_block = x
        r_pos = xyz_block_to_pos(r_block)
        p_pos = xyz_block_to_pos(p_block)
        ts_pos = xyz_block_to_pos(ts_block)
        r_noised, p_noised, ts_noised = add_noise_to_distance_qst(
                r_pos, p_pos, ts_pos, sigma, steps=steps, patience=patience, device=device
                )
        r_noised = reform_xyz_block(r_block, r_noised)
        p_noised = reform_xyz_block(p_block, p_noised)
        ts_noised = reform_xyz_block(ts_block, ts_noised)
        out = (r_noised, p_noised, ts_noised)
        return out

    if not os.path.isdir(f"result_B_fix/sigma_{sigma:0.1e}"):
        os.system(f"mkdir result_B_fix/sigma_{sigma:0.1e}")

    with mp.Pool(16) as p:
        ret = p.map(func, list(zip(r_blocks, p_blocks, ts_blocks))[:100])
    #ret = []
    #for x in list(zip(r_blocks, p_blocks, ts_blocks)):
    #    ret.append(func(x))

    r_aug_blocks = [o[0] for o in ret]
    p_aug_blocks = [o[1] for o in ret]
    ts_aug_blocks = [o[2] for o in ret]

    r_msg = "\n$$$$$\n".join(r_aug_blocks) + "\n$$$$$"
    p_msg = "\n$$$$$\n".join(p_aug_blocks) + "\n$$$$$"
    ts_msg = "\n$$$$$\n".join(ts_aug_blocks) + "\n$$$$$"
    with open(f"result_B_fix/sigma_{sigma:0.1e}/b97d3_r_test_B.xyz","w") as f:
        f.write(r_msg)
    with open(f"result_B_fix/sigma_{sigma:0.1e}/b97d3_p_test_B.xyz","w") as f:
        f.write(p_msg)
    with open(f"result_B_fix/sigma_{sigma:0.1e}/b97d3_ts_test_B.xyz","w") as f:
        f.write(ts_msg)

