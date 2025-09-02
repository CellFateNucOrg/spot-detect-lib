#!/bin/bash
#SBATCH --job-name=SPOT_det
#SBATCH --output=SPOT_det_%j.out          # Errors files will appear in directory with this script
#SBATCH --error=SPOT_det_%j.err
#SBATCH --time=1:30:00                    # Max runtime (hh:mm:ss)
#SBATCH --mem=124G                        # Memory requested (adjust depending on dataset size)
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task (parallel processes)
#SBATCH --ntasks=1
#SBATCH --partition=all


##### Set all your directories here #####
# --- Only the RAW directory and BASE directory need to be set manually ---
# RAW_DIR  = folder where your input images are stored
# BASE_DIR = main working directory where all output folders will be created
#             (if you already used the nuclei segmentation pipeline, use the same BASE_DIR)
#             This will help keep results organized in one place.
#
# Example:
# RAW_DIR=/path/to/my/images
# BASE_DIR=/path/to/my/segmentation/project

##### Additional notes for users #####
# - Adjust memory (--mem) and runtime (--time) according to the size and number of your images.
#   For large datasets, you may need much more memory (e.g., 256G) and longer runtime.
#
# - The number of CPUs (--cpus-per-task) corresponds to parallel processes.
#   Increasing CPUs will speed up the analysis but also require more memory.


# ——— User parameters Example———

RAW_DIR="/mnt/external.data/MeisterLab/mvolosko/image_project/final_testing/all_ds_test/raw_images/"
BASE_DIR="/mnt/external.data/MeisterLab/mvolosko/image_project/final_testing/all_ds_test/"


# ----- The Processing ------ 

# Output directories exist
INPUT_DIR="${RAW_DIR}"
NUCLEI_MASK_DIR="${BASE_DIR}/segmentation/"
NUCLEI_CSV_DIR="${BASE_DIR}/nuclei/"
OUTPUT_DIR="${BASE_DIR}/spots/"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$BASE_DIR"

#run with apptainer

CONTAINER=/mnt/external.data/MeisterLab/mvolosko/spot_detection_environment/sponuc.sif
ENV_PATH=/mnt/external.data/MeisterLab/mvolosko/spot_detection_environment/sponuc-env
PROJECT_PATH=/mnt/external.data/MeisterLab/

# Run Python from micromamba env, bind both env and project
# Binding is important in order for container to find the location
apptainer exec \
    --bind $ENV_PATH:/mnt/env \
    --bind $PROJECT_PATH \
    $CONTAINER \
    /mnt/env/bin/python /mnt/external.data/MeisterLab/mvolosko/image_project/spot-detect-lib/spot_detection/main.py \
      --base-dir        "$BASE_DIR" \
      --input-dir       "$INPUT_DIR" \
      --nuclei-mask-dir "$NUCLEI_MASK_DIR" \
      --nuclei-csv-dir  "$NUCLEI_CSV_DIR" \
      --output-dir      "$OUTPUT_DIR"


# # CONDA ENV - OLD version which is not reproducible
# If you create your own conda environment, activate it like this:
# source /home/mvolosko/miniconda3/etc/profile.d/conda.sh
# conda activate lhcellpose

# # run the script
# python -u ../spot_detection/main.py \
#   --base-dir        "$BASE_DIR" \
#   --input-dir       "$INPUT_DIR" \
#   --nuclei-mask-dir "$NUCLEI_MASK_DIR" \
#   --nuclei-csv-dir  "$NUCLEI_CSV_DIR" \
#   --output-dir      "$OUTPUT_DIR" 
# conda deactivate
