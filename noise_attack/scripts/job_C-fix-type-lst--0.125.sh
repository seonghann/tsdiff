#PBS -N C-fix-type-lst--0.125
#PBS -l nodes=cnode8:ppn=16
#PBS -l walltime=500:00:00

cd $PBS_O_WORKDIR
echo `cat $PBS_NODEFILE`
cat $PBS_NODEFILE
NPROCS=`wc -l < $PBS_NODEFILE`

date
export PYTHONPATH=~/MolDiff:$PYTHONPATH

python3 method_C_fix.py --sigma -0.125
