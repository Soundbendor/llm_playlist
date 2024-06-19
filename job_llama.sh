#!/bin/bash
#SBATCH -p dgx2
#SBATCH -c 8
#SBATCH --gres=gpu:4
#SBATCH --mem=500G
#SBATCH --job-name=llama500
#SBATCH -t 2-00:00:00
#SBATCH --export=ALL
#SBATCH -o llama500.out
#SBATCH -e llama500.err

module load openssl/1.1.1w python/3.12 cuda/12.2

# activate env
source env/bin/activate

export HF_HOME=/nfs/guille/eecs_research/soundbendor/zontosj/.cache/

python pairwise_rerank.py