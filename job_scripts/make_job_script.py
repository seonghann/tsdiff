

def create_msg(job_name, node, config_file_path, ppn=4, run_file_path="train.py", **kwargs):
    msg = f"""#!/bin/bash
#PBS -N {job_name}
#PBS -l nodes=gnode{node}:ppn={ppn}
#PBS -l walltime=500:00:00
"""
    msg += """
cd $PBS_O_WORKDIR
echo `cat $PBS_NODEFILE`
cat $PBS_NODEFILE
NPROCS=`wc -l < $PBS_NODEFILE`

for DEVICEID in `seq 0 7`; do
    AVAILABLE=`nvidia-smi -i ${DEVICEID} | grep "No" | wc -l`
    if [ ${AVAILABLE} == 1 ] ; then
        break;
    fi

done
date
source ~/.bashrc
conda deactivate
conda activate geodiff
export PYTHONPATH=~/MolDiff:$PYTHONPATH
echo $DEVICEID

"""
    msg += f"""
CUDA_VISIBLE_DEVICES=$DEVICEID python3 {run_file_path} {config_file_path} --fn _ """
    for k, v in kwargs.items():
        msg += f"--{k} {v} "
    return msg

if __name__ == "__main__":
    from itertools import cycle
    config_file_list = []
    _ = ["ts_dv3_newedge_base", "ts_dv3_newedge_nolocal"]
    config_file_list += ["configs/exp9/" + x + ".yml" for x in _]
    job_name_list = ["exp9_base", "exp9_nolocal"]
    nodes = cycle([8])
    
    # kwargs
    #project_list = ["TS"] * 9 + ["Drug"] * 9
    project_list = ["TS-geom-exp9"] * len(config_file_list)
    name_list = job_name_list
    kwargs_list = [{"project":p, "name":n} for p, n in zip(project_list, name_list)]
    
    for cfg_fn, jn, node, kwargs in zip(config_file_list, job_name_list, nodes, kwargs_list):
        msg = create_msg(jn, node, cfg_fn, **kwargs)
        with open(f"{jn}.sh", "w") as f: f.write(msg)
