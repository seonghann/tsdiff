import torch
import pickle
import os
import argparse
import glob


def read_xyz_file(fn, sep="\t"):
    if os.path.isfile(fn):

        with open(fn, "r") as f:
            lines = f.readlines()

        line_sep = ""
        for l in lines:
            if "$$$$$" in l:
                line_sep = "$$$$$"
                break
        if not line_sep:
            lines = "".join(lines).strip()
            lines = lines.split("\n")
            cnt = 0
            xyz_blocks = []
            for i, l in enumerate(lines):
                try:
                    num = int(l.strip())
                    num += 2
                    a = lines[cnt: cnt + num]
                    xyz_blocks.append("\n".join(a).strip())
                    cnt += num

                except:
                    pass

        else:
            xyz_blocks = "".join(lines).strip()
            xyz_blocks = [xyz.strip() for xyz in xyz_blocks.split(line_sep)]
            xyz_blocks = [xyz for xyz in xyz_blocks if xyz]

        xyz_list = [
            torch.Tensor([
                [float(v.strip()) for v in l.strip().split(sep)[1:]]
                for l in xyz.split("\n")[2:]
            ]) for xyz in xyz_blocks
        ]
        return xyz_list, fn

    elif os.path.isdir(fn):
        fn_ = os.path.join(fn, "*.xyz")
        files = glob.glob(fn_)
        try:
            files.sort(key=lambda x : int(x.split(".")[-2].split("_")[-1]))
        except:
            pass

        xyz_list = []
        for f_ in files:
            with open(f_, "r") as f:
                xyz = [float(l.split(sep)[1:].split()) for l in f.readlines()[2:]]
                xyz_list.append(torch.Tensor(xyz))
        return xyz_list, files

    else:
        raise IOError(f"{fn} is not a xyz file nor a directory of xyz files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    str2bool = lambda x: True if x.lower() == "true" else False
    parser.add_argument("pickle_file", type=str,
        help="original data file. eg) data/TS/b97d3/random_split3/test_data.pkl"
    )
    parser.add_argument("--guess_ts", "-guess_ts", type=str, default=None,
        help="path of connected xyz file of ts_guess or directory of ts xyz files"
    )
    parser.add_argument("--reactants", "-reactants", type=str, default=None,
        help="path of connected xyz file of reactants or directory of ts xyz files"
    )
    parser.add_argument("--products", "-products", type=str, default=None,
        help="path of connected xyz file of products or directory of ts xyz files"
    )
    parser.add_argument("--sep", "-sep", type=str, default="\t",
        help="separation token of single line of xyz file."
             "eg) N<sep>1.11324<sep>4.22231<sep>3.22503\n"
                 "C<sep>2.33245<sep>1.22231<sep>1.12345\n"
                 "..."
    )
    parser.add_argument("--save","-save",type=str, default=None,
        help="save path of the post-processed data file,"
             "if not sepcified, overwrite original data file."
    )
    parser.add_argument("--f", "-f", type=str2bool, default=False,
        help="In case of specified save path already exist, force to overwrite."
    )
    args = parser.parse_args()

    with open(args.pickle_file, "rb") as f:
        data = pickle.load(f)

    if args.guess_ts is not None:
        xyz_list, _ = read_xyz_file(args.guess_ts, sep=args.sep)
        assert len(data) == len(xyz_list)
        for d, xyz in zip(data, xyz_list):
            d.ts_guess = xyz
        print(f"Update ts_guess xyz from file {args.ts_guess}")

    if args.reactants is not None:
        xyz_list, _ = read_xyz_file(args.reactants, sep=args.sep)
        print(len(data), len(xyz_list))
        assert len(data) == len(xyz_list)
        for d, xyz in zip(data, xyz_list):
            d.pos_r = xyz
        print(f"Update reactants xyz from file {args.reactants}")

    if args.products is not None:
        xyz_list, _ = read_xyz_file(args.products, sep=args.sep)
        assert len(data) == len(xyz_list)
        for d, xyz in zip(data, xyz_list):
            d.pos_p = xyz
        print(f"Update products xyz from file {args.products}")

    if args.save is None:
        save_path = args.pickle_file
        print(f"save path is not set. overwrite given pickle file {args.pickle_file}")

    else:
        save_path = args.save
        print(f"save path is set as {args.save}")
        if os.path.isfile(args.save):
            raise IOError(f"...But file exist, {args.save}"
                           "Cannot overwrite exist file")

    with open(save_path, "wb") as f:
        pickle.dump(data, f)
