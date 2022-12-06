#!/bin/bash
#PBS -N sampling_newedge_nolocal
#PBS -l nodes=gnode8:ppn=4
#PBS -l walltime=500:00:00

cd $PBS_O_WORKDIR
echo `cat $PBS_NODEFILE`
cat $PBS_NODEFILE
NPROCS=`wc -l < $PBS_NODEFILE`

for DEVICEID in `seq 0 3`; do
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


CUDA_VISIBLE_DEVICES=1 python3 test_ts.py logs/ts_dv3_newedge_nolocal___dv3_newedge_nolocal/checkpoints/181000.pt --start_idx 0 --end_idx 1000 
