import torch
from GeoDiff.noise_attack.D2X import get_distance_matrix, get_coord_from_ditance


def xyz_block_to_pos(block, dtype=torch.float):
    lines = block.strip().split("\n")
    xyz = []
    for l in lines[2:]:
        a_xyz = [float(v) for v in l.strip().split("\t")[1:]]
        xyz.append(a_xyz)
    return torch.DoubleTensor(xyz).to(dtype)

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
    
    str2bool = lambda x : True if x=="true" or x=="True" else False
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma","-sigma",required=True,type=float)
    parser.add_argument("--r_path", "-r_path", 
            default="/home/ksh/MolDiff/tsdiff/"
                    "data/TS/b97d3/random_split/"
                    "raw_data/b97d3_r_test.xyz", 
            type=str)
    parser.add_argument("--p_path", "-p_path", 
            default="/home/ksh/MolDiff/tsdiff"
                    "/data/TS/b97d3/random_split/"
                    "raw_data/b97d3_p_test.xyz", 
            type=str)
    parser.add_argument("--ts_path", "-ts_path", 
            default="/home/ksh/MolDiff/tsdiff/"
                    "data/TS/b97d3/random_split/"
                    "raw_data/b97d3_ts_test.xyz", 
            type=str)
    parser.add_argument("--save_type", "-save_type", 
            default="pkl", 
            choices=["pkl", "xyz"])
    parser.add_argument("--save_prefix","-save_prefix",
            default="b97d3_random_split",
            type=str)
    
    parser.add_argument("--feat_dict", "-feat_dict",
            default="../data/TS/b97d3/random_split/feat_dict.pkl",
            type=str)
    
    parser.add_argument("--random", "-random", default=True, type=str2bool,
            help="Deprecated")
    parser.add_argument("--n","-n",default=1,type=int,
            help="Deprecated")
    parser.add_argument("--steps","-steps",default=3000,type=int,
            help="Deprecated")
    parser.add_argument("--patience","-patience",default=300,type=int,
            help="Deprecated")
    args = parser.parse_args()
    
    # read reactant & product test dataset to add noise (xyz file)
    r_blocks = read_xyz_file(args.r_path)
    p_blocks = read_xyz_file(args.p_path)
    ts_blocks = read_xyz_file(args.ts_path)
    
    device = "cpu"
    def func(x):
        r_block, p_block, ts_block = x
        r_pos = xyz_block_to_pos(r_block, dtype=torch.double)
        p_pos = xyz_block_to_pos(p_block, dtype=torch.double)
        ts_pos = xyz_block_to_pos(ts_block, dtype=torch.double)

        out = []
        for i in range(args.n):
            if args.save_type == "pkl":
                ts_noised = ts_pos + torch.randn(ts_pos.size()) * args.sigma 
                ts_noised = reform_xyz_block(ts_block, ts_noised)
                out.append((ts_noised,))

            else:
                r_noised = r_pos + torch.randn(r_pos.size()) * args.sigma
                p_noised = p_pos + torch.randn(p_pos.size()) * args.sigma
                r_noised = reform_xyz_block(r_block, r_noised)
                p_noised = reform_xyz_block(p_block, p_noised)
                out.append((r_noised, p_noised,))
        return out
    
    # generate nosied coordinate
    with mp.Pool(16) as p:
        ret = p.map(func, list(zip(r_blocks, p_blocks, ts_blocks)))
       
    if args.save_type == "pkl":
        noised_data_iter = [[o[i][0] for o in ret] for i in range(args.n)]
    else:
        r_data_iter = [[o[i][0] for o in ret] for i in range(args.n)]
        p_data_iter = [[o[i][1] for o in ret] for i in range(args.n)]
        noised_data_iter = zip(r_data_iter, p_data_iter)


    save_dir = (f"{args.save_prefix}_cartesian/"
                f"{args.save_type}_data/"
                f"sigma_{args.sigma:0.1e}")
    _ = save_dir.split("/")
    _ = ["/".join(_[:i]) for i in range(1, len(_)+1)]
    for p in _:
        if not os.path.isdir(p):
            os.system(f"mkdir {p}")

    # save
    if args.save_type == "pkl":
        r_smarts = [b.split("\n")[1].strip() for b in r_blocks]
        p_smarts = [b.split("\n")[1].strip() for b in p_blocks]
        import pickle
        from tsdiff.utils.datasets import generate_ts_data2

        with open(args.feat_dict, "rb") as f:
            feat_dict = pickle.load(f)
        data_list = []
        
        for i, aug_blocks in enumerate(noised_data_iter):
            for xyz_r, xyz_p, xyz_gt, xyz_aug in zip(r_blocks, p_blocks, ts_blocks, aug_blocks):
                r = xyz_r.split("\n")[1].strip()
                p = xyz_p.split("\n")[1].strip()
                data, _ = generate_ts_data2(r, p, xyz_gt, feat_dict=feat_dict)
                data.ts_guess = xyz_block_to_pos(xyz_aug, dtype=torch.float)
                data.pos_r = xyz_block_to_pos(xyz_r, dtype=torch.float)
                data.pos_p = xyz_block_to_pos(xyz_p, dtype=torch.float)
                data_list.append(data)

            num_cls = [len(v) for k, v in feat_dict.items()]
            for data in data_list:
                feat_onehot = []
                feats = data.r_feat.T
                for feat, n_cls in zip(feats, num_cls):
                    feat_onehot.append(torch.nn.functional.one_hot(feat, num_classes=n_cls))
                data.r_feat = torch.cat(feat_onehot, dim= -1)
                
                feat_onehot = []
                feats = data.p_feat.T
                for feat, n_cls in zip(feats, num_cls):
                    feat_onehot.append(torch.nn.functional.one_hot(feat, num_classes=n_cls))
                data.p_feat = torch.cat(feat_onehot, dim= -1)
            

            with open(f"{save_dir}/data_{i}.pkl", "wb") as f:
                pickle.dump(data_list, f)
    
    elif args.save_type =="xyz":
        for i, (r, p) in enumerate(noised_data_iter):
            r_msg = "\n$$$$$\n".join(r) + "\n$$$$$"
            p_msg = "\n$$$$$\n".join(p) + "\n$$$$$"
            #ts_msg = "\n$$$$$\n".join(ts) + "\n$$$$$"
            with open(f"{save_dir}/r_aug_{i}.xyz","w") as f:
                f.write(r_msg)
            with open(f"{save_dir}/p_aug_{i}.xyz","w") as f:
                f.write(p_msg)
            #with open(f"{save_dir}/ts_aug_{i}.xyz","w") as f:
            #    f.write(ts_msg)
