#!/bin/bash
#SBATCH --job-name=spot_analysis
#SBATCH --output=spot_analysis_%j.out
#SBATCH --error=spot_analysis_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1       
#SBATCH --partition=all
#SBATCH --ntasks=2


# Load Conda (adjust this if your system uses a different path)
source /home/mvolosko/miniconda3/etc/profile.d/conda.sh

# Activate the environment
conda activate lhcellpose

python /mnt/external.data/MeisterLab/mvolosko/image_project/spotDetection/code/utils/post_processing.py

conda deactivate