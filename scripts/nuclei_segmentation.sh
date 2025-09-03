#!/usr/bin/env bash
#SBATCH --job-name=NUCLEI_SEGM
#SBATCH --output=NUCLEI_SEGM_%j.out
#SBATCH --error=NUCLEI_SEGM_%j.err
#SBATCH --time=01:00:00                    # Max runtime (hh:mm:ss)
#SBATCH --mem=64G                          # Memory requested (increase if using high-res images)
#SBATCH --cpus-per-task=1                  # Number of CPU cores per task
#SBATCH --gres=gpu:1                       # Request 1 GPU (required for speed; job may work without GPU but slower and less tested)

###SBATCH --partition=all -w izbdodoma # Uncomment if cluster don't allocate you to GPU node
set -euo pipefail


##### Notes for users #####
# - GPU requirement:
#   This script is optimized for GPU usage. It *might* run without GPU (by removing the `--gres=gpu:1` line),
#   but the runtime will be significantly longer and correctness/stability is less tested in CPU-only mode.
#
# - Memory:
#   High-resolution images can consume very large amounts of memory.
#   If you run into "Out of Memory" errors, increase `--mem` (e.g., 124G or more).
#


##### User parameters (to set manually) #####
# RAW_DIR      = Path to folder with raw input images
# DENOISED_DIR = Optional path to denoised images (or set to "None")
# Path to pretrained segmentation model (does not change across runs)
# MODEL_DIR    = "/mnt/external.data/MeisterLab/lhinder/segmentation_3d_anja/code/worms_1000epochs_v0" 
# OUT_ROOT     = Path to output directory where results will be stored

# Example configuration:
RAW_DIR="/mnt/external.data/MeisterLab/mvolosko/image_project/final_testing/all_ds_test/raw_images/"
DENOISED_DIR="None"
MODEL_DIR="/mnt/external.data/MeisterLab/lhinder/segmentation_3d_anja/code/worms_1000epochs_v0"
OUT_ROOT="/mnt/external.data/MeisterLab/mvolosko/image_project/final_testing/all_ds_test/"

# Or high resolutoin images

# RAW_DIR="/mnt/external.data/MeisterLab/mvolosko/image_project/high_res_images/1273/"
# DENOISED_DIR="None"
# MODEL_DIR="/mnt/external.data/MeisterLab/lhinder/segmentation_3d_anja/code/worms_1000epochs_v0"
# OUT_ROOT="/mnt/external.data/MeisterLab/mvolosko/image_project/SDC1/1273_fast_imaging/"

# --- Container setup ---
CONTAINER=/mnt/external.data/MeisterLab/mvolosko/spot_detection_environment/sponuc.sif
ENV_PATH=/mnt/external.data/MeisterLab/mvolosko/spot_detection_environment/sponuc-env
PROJECT_PATH=/mnt/external.data/MeisterLab/
SCRIPTS=/mnt/external.data/MeisterLab/mvolosko/image_project/spot-detect-lib/segmentation
# make sure output root exists
mkdir -p "${OUT_ROOT}"

cd ../

# If you want to segment nd2 images, please change --pattern to ".nd2"
# ——— 1. Segment & EDT ———
apptainer exec --nv \
    --bind $ENV_PATH:/mnt/env \
    --bind $PROJECT_PATH \
    --bind $SCRIPTS \
    $CONTAINER \
    /mnt/env/bin/python -m \
    segmentation.cli segment \
        --raw-dir      "${RAW_DIR}" \
        --denoised-dir "${DENOISED_DIR}" \
        --pattern      "*.tif" \
        --model        "${MODEL_DIR}" \
        --out-root     "${OUT_ROOT}" \
        --gpu \
        --do-qc \
        --blur-sigma "0"  # \ 2.0 for high-res
        #--diameter "125" #also for high resolution

# ——— 2. Distance–intensity analysis & summarization ———
apptainer exec \
    --bind $ENV_PATH:/mnt/env \
    --bind $PROJECT_PATH \
    $CONTAINER \
    /mnt/env/bin/python -m segmentation.cli analyze \
        --raw-dir  "${RAW_DIR}" \
        --seg-dir "${OUT_ROOT}/segmentation" \
        --out-root "${OUT_ROOT}"

echo "✅ Pipeline completed – results in ${OUT_ROOT}"



# Old way
# cd ../
# # activate env
# set +u
# source /home/mvolosko/miniconda3/etc/profile.d/conda.sh
# conda activate lhcellpose
# set -u
# # make sure output root exists
# mkdir -p "${OUT_ROOT}"

# # ——— 1. Segment & EDT (use raw as fallback if no denoised) ———
# # python -m segmentation.cli segment \
# #     --raw-dir      "${RAW_DIR}" \
# #     --denoised-dir "${DENOISED_DIR}" \
# #     --pattern      "*.nd2" \
# #     --model        "${MODEL_DIR}" \
# #     --out-root     "${OUT_ROOT}" \
# #     --gpu \
# #     --do-qc

# # with no denoising step:
# python -m segmentation.cli segment \
#     --raw-dir  "${RAW_DIR}" \
#     --denoised-dir "${DENOISED_DIR}" \
#     --pattern  "*.tif" \
#     --model    "${MODEL_DIR}" \
#     --out-root "${OUT_ROOT}" \
#     --gpu \
#     --do-qc \
#     --blur-sigma "2.0"

# # ——— 2. Distance–intensity analysis & summarization ———
# python -m segmentation.cli analyze \
#     --raw-dir  "${RAW_DIR}" \
#     --seg-dir  "${OUT_ROOT}/segmentation" \
#     --out-root "${OUT_ROOT}"

# echo "✅ Pipeline completed – results in ${OUT_ROOT}"
