import pickle
import rdkit
import rdkit.Chem as Chem
import os
import argparse

atomicnum_to_symbol = {}
atomicnum_to_symbol[6] = "C"
atomicnum_to_symbol[7] = "N"
atomicnum_to_symbol[8] = "O"
atomicnum_to_symbol[1] = "H"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-path", type=str, required=True)
    args = parser.parse_args()

    args.file_name = args.path.split("/")[-1]
    args.data_dir = os.path.join(*args.path.split("/")[:-1])
    print(f"pck file name : {args.file_name}")
    print(f"pck file dir : {args.data_dir}")

    data_list = []
    with open(os.path.join(args.data_dir, args.file_name), "rb") as f:
        data_list = pickle.load(f)

    save_dir = os.path.join(args.data_dir, "xyz_gen")  # f"samples_{j}")
    os.system(f"mkdir {save_dir}")
    for j, data in enumerate(data_list):
        # title = data.smiles
        title = f"samples_{j}"
        pos = data.pos
        pos_gen = data.pos_gen.view(-1, pos.size(0), pos.size(1))
        pos_ref = data.pos.view(-1, pos.size(0), pos.size(1))

        symbols = [atomicnum_to_symbol[a] for a in data.atom_type.tolist()]

        for i, pos_ in enumerate(pos_gen):
            # msg = f"{data.smiles}_{i}\n\n"
            msg = f"{data.atom_type.shape[0]}\n\n"
            for s, xyz in zip(symbols, pos_):
                msg += f"{s} "
                for u in xyz:
                    msg += f"{u} "
                msg += "\n"

            with open(f"{save_dir}/{title}_{i}.xyz", "w") as f:
                f.write(msg)

        msg = f"{data.atom_type.shape[0]}\n\n"
        for s, xyz in zip(symbols, pos):
            msg += f"{s} "
            for u in xyz:
                msg += f"{u} "
            msg += "\n"

        with open(f"{save_dir}/{title}_ref.xyz", "w") as f:
            f.write(msg)
