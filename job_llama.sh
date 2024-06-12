#!/bin/bash
#SBATCH -p preempt
#SBATCH -c 6
#SBATCH --gres=gpu:4
#SBATCH --mem=100G
#SBATCH --job-name=llama100test
#SBATCH -t 1-00:00:00
#SBATCH --export=ALL
#SBATCH -o llama100test.out
#SBATCH -e llama100test.err

module load openssl/1.1.1w python/3.12 cuda/12.2

# activate env
source env/bin/activate

export HF_HOME=/nfs/guille/eecs_research/soundbendor/zontosj/.cache/

python pairwise_rerank.py