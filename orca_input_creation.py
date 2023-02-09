import torch
import pickle
import argparse
import os

SYMBOL_DICT = {}
SYMBOL_DICT[6] = "C"
SYMBOL_DICT[7] = "N"
SYMBOL_DICT[8] = "O"
SYMBOL_DICT[1] = "H"

def xyz_string_to_orca_input(xyz_str, calc_method, charge=0, spin=1):
    xyz_lines = xyz_str.strip().split("\n")[2:]
    xyz_lines = [calc_method, f"* XYZ {charge} {spin}"] + xyz_lines + ["*"]
    orca_input = "\n".join(xyz_lines)
    return orca_input

def tensor_to_xyz(pos, atom, title=""):
    assert len(pos) == len(atom)
    if isinstance(atom, torch.Tensor):
        atom = [SYMBOL_DICT[int(i)] for i in atom]
    lines = [f"{len(atom)}", title,]
    for a, p in zip(atom, pos):
        line = [a] + [str(x.item()) for x in list(p)]
        line = "\t".join(line)
        lines.append(line)
    xyz = "\n".join(lines)
    return xyz

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="samples.all file path")
    parser.add_argument("--save_dir", type=str, required=True, help="e.g orca/model_swish/sample100_seed1/")
    parser.add_argument("--start_idx", type=int, default=0, help="data[start_idx:end_idx]")
    parser.add_argument("--end_idx", type=int, default=99999, help="data[start_idx:end_idx]")
    parser.add_argument("--job_name", type=str, default="orca_ts_calc", help="PBS job name")
    parser.add_argument("--avail_nodes", type=int, default=[11,12,13,14], nargs="+", help="currently available cnode index, 11 12 13 14...")
    args = parser.parse_args()
    import itertools
    
    if not os.path.isdir(args.save_dir):
        os.system(f"mkdir -p {args.save_dir}")
    
    calc_method = "! wB97X-D3 def2-TZVP OptTS NumFreq"
    with open(args.data_path,"rb") as f:
        data = pickle.load(f)

    nodes = itertools.cycle(args.avail_nodes)
    
    for i, d in enumerate(data[args.start_idx:args.end_idx]):
        if i % 2 != 0: continue
        pos, atom = d.pos_gen, d.atom_type
        if pos.dim() == 3: pos = pos[-1]
        xyz = tensor_to_xyz(pos, atom)
        inp_str = xyz_string_to_orca_input(xyz, calc_method)

        save_dir = os.path.join(args.save_dir,f"sample_{i}")
        os.system(f"mkdir -p {save_dir}")
        # orca input generation
        with open(f"{save_dir}/input","w") as f:
            f.write(inp_str)

        # job script
        job_script_msg = f"""#!/bin/bash
#PBS -N {args.job_name}_{i}
#PBS -l nodes=cnode{next(nodes)}:ppn=4
#PBS -l walltime=500:00:00
    """ + """
cd $PBS_O_WORKDIR
echo `cat $PBS_NODEFILE`
cat $PBS_NODEFILE
NPROCS=`wc -l < $PBS_NODEFILE`

date
source ~/.bashrc
conda deactivate
conda activate tsdiff
export PYTHONPATH=~/MolDiff/tsdiff:$PYTHONPATH
""" + f"""
/appl/orca_4.2.1/orca_4_2_1_linux_x86-64_openmpi314/orca {save_dir}/input > {save_dir}/log"""
        jn = f"job_{i}.sh"
        with open(f"{os.path.join(args.save_dir, jn)}", "w") as f:
            f.write(job_script_msg)

