#!/bin/bash
#SBATCH -w cn-m-2
#SBATCH -p soundbendor
#SBATCH -A soundbendor
#SBATCH --job-name=llm_playlist
#SBATCH -t 4-00:00:00
#SBATCH -c 1
#SBATCH --mem=90G
#SBATCH --export=ALL

# activate env
source env/bin/activate

python post_llm.py
