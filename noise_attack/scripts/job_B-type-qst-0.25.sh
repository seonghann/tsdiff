#PBS -N B-type-qst-0.25
#PBS -l nodes=cnode13:ppn=16
#PBS -l walltime=500:00:00

cd $PBS_O_WORKDIR
echo `cat $PBS_NODEFILE`
cat $PBS_NODEFILE
NPROCS=`wc -l < $PBS_NODEFILE`

date
conda deactivate
conda activate geodiff
export PYTHONPATH=~/MolDiff:$PYTHONPATH

python3 method_B.py --sigma 0.25
