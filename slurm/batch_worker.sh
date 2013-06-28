#!/bin/sh

#SBATCH --time 10
#SBATCH --partition=intel
#SBATCH --mem-per-cpu=1G

# JOB OUTPUT
#
#SBATCH --workdir=/home/bartosz/slurm
#SBATCH --output=logs/params_scan.%N.j%j
#SBATCH --error=logs/params_scan.%N.j%j.error

FI=$((TASK_ID/26+5)) 
FE=$((TASK_ID%26+5))
PARAMS="--input-process white --ge_sigma $1 --gi_sigma $2"
OUTFNAME="{input_process}_noise_fe{fe}_fi{fi}_se{ge_sigma}_si{gi_sigma}.pickle"
OUTPATH=/home/bartosz/slurm/results/params_scan/
SCRIPT=/home/bartosz/repos/projects/transfer_functions/params_scan.py

CMD="$SCRIPT $FE $FI $OUTPATH $PARAMS --pattern $OUTFNAME"
echo "running python" $CMD
python $CMD
