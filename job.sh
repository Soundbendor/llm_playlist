#!/bin/bash
#SBATCH -p preempt
#SBATCH --job-name=process
#SBATCH -t 4-00:00:00
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH --export=ALL

# activate env
source env/bin/activate

python process_gpt_output.py