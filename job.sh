#!/bin/bash
#SBATCH -w cn-d-1
#SBATCH -p preempt
#SBATCH --job-name=popularity_metrics
#SBATCH -t 4-00:00:00
#SBATCH -c 1
#SBATCH --mem=90G
#SBATCH --export=ALL

# activate env
source env/bin/activate

python calc_new_metrics.py