#!/bin/bash
#PBS -N exp8_base
#PBS -l nodes=gnode2:ppn=4
#PBS -l walltime=500:00:00

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


CUDA_VISIBLE_DEVICES=3 python3 train_ts.py configs/exp8/ts_dv3_newedge_base.yml --fn _ --project TS-geom-exp8 --name exp8_base 
