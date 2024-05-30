#!/bin/bash
#SBATCH -p preempt
#SBATCH -w cn-d-1
#SBATCH --job-name=pop
#SBATCH -t 4-00:00:00
#SBATCH -c 1
#SBATCH --mem=180G
#SBATCH --export=ALL

#SBATCH -o calc_new_metrics.out
#SBATCH -e calc_new_metrics.err

# activate env
source env/bin/activate

python calc_new_metrics.py