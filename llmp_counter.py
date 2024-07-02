#!/bin/bash
#SBATCH -p soundbendor
#SBATCH -A soundbendor
#SBATCH -t 3-00:00:00
#SBATCH --mem=1G
#SBATCH --job-name=llmcount
#SBATCH --export=ALL

python playlist_song_counts.py

