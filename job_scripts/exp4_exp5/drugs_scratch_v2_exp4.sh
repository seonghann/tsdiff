#!/bin/bash
#PBS -N drugs_scratch_v2_exp4
#PBS -l nodes=gnode8:ppn=4
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


CUDA_VISIBLE_DEVICES=$DEVICEID python3 train.py configs/exp4/drugs_scratch_v2_exp4.yml --fn _ --project TS-geom-exp4 --name drugs_scratch_v2_exp4 