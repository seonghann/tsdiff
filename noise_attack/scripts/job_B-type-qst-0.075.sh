#PBS -N B-type-qst-0.075
#PBS -l nodes=cnode6:ppn=16
#PBS -l walltime=500:00:00

cd $PBS_O_WORKDIR
echo `cat $PBS_NODEFILE`
cat $PBS_NODEFILE
NPROCS=`wc -l < $PBS_NODEFILE`

date
conda deactivate
conda activate geodiff
export PYTHONPATH=~/MolDiff:$PYTHONPATH

python3 method_B.py --sigma 0.075
