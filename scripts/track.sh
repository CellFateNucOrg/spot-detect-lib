#!/bin/bash
#SBATCH --job-name=tracking
#SBATCH --output=TRACK_%j.out
#SBATCH --error=TRACK_%j.err
#SBATCH --time=3:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --partition=all


INPUT_DIR=""
OUTPUT_DIR=""

# make sure the output directories exist
mkdir -p "$OUTPUT_DIR"


# load your conda env
source /home/mvolosko/miniconda3/etc/profile.d/conda.sh
conda activate lhcellpose


cd ../spot_tracking/

python -u tracking.py \
  --output "$OUTPUT_DIR" \
  --spot_dir "$INPUT_DIR/spots/count_spots/" \
  --segmentation_dir "$INPUT_DIR/segmentation/"