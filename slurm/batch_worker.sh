#!/bin/sh

#SBATCH --time 10
#SBATCH --partition=intel
#SBATCH --mem-per-cpu=1G

# JOB OUTPUT
#
#SBATCH --workdir=/home/bartosz/slurm
#SBATCH --output=logs/batch_worker.%N.j%j
#SBATCH --error=logs/batch_worker.%N.j%j.error

FE=$((TASK_ID/20+1)) 
FI=$((TASK_ID%20+1))
OUTPATH=/home/bartosz/slurm/results/params_scan/
SCRIPT=/home/bartosz/repos/projects/transfer_functions/params_scan.py

echo "running" $SCRIPT $FE $FI $OUTPATH
python $SCRIPT $FE $FI $OUTPATH
