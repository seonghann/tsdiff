def get_pbs_msg(job_name, node):
    msg = f"""#PBS -N {job_name}
#PBS -l nodes=cnode{node}:ppn=16
#PBS -l walltime=500:00:00
"""
    return msg


def get_env_msg():
    msg = """
cd $PBS_O_WORKDIR
echo `cat $PBS_NODEFILE`
cat $PBS_NODEFILE
NPROCS=`wc -l < $PBS_NODEFILE`

date
export PYTHONPATH=~/MolDiff:$PYTHONPATH
"""
    return msg


def get_run_msg(sigma):
    msg = f"""
python3 add_rxn_coords_noise.py --sigma {sigma} --save_type xyz --random False --interpolation qst --n 1 --r_path /home/ksh/MolDiff/tsdiff/data/TS/b97d3/random_split/raw_data/b97d3_r_train.xyz --p_path /home/ksh/MolDiff/tsdiff/data/TS/b97d3/random_split/raw_data/b97d3_p_train.xyz --ts_path /home/ksh/MolDiff/tsdiff/data/TS/b97d3/random_split/raw_data/b97d3_ts_train.xyz --save_prefix b97d3_random_split_train
#"""
    return msg


sigmas = [-0.25, -0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20, 0.25]
job_name = [f"qst-noised-rp-train-{sigma:0.2f}" for sigma in sigmas]
for i, (s, n) in enumerate(zip(sigmas, job_name)):
    msg = get_pbs_msg(n, i + 4)
    msg += get_env_msg()
    msg += get_run_msg(s)
    with open(f"job_{n}.sh", "w") as f:
        f.write(msg)
