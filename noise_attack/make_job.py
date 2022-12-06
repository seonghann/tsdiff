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
python3 method_C_fix.py --sigma {sigma}
"""
    return msg

#sigmas = [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25,
sigmas = [-0.025, -0.05, -0.075, -0.10, -0.125, -0.15, -0.175, -0.2, -0.225, -0.25,]

#sigmas = [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
job_name = [f"C-fix-type-lst-{sigma}" for sigma in sigmas]
for i, (s, n) in enumerate(zip(sigmas, job_name)):
    msg = get_pbs_msg(n, i+4)
    msg += get_env_msg()
    msg += get_run_msg(s)
    with open(f"job_{n}.sh","w") as f:
        f.write(msg)

