#!/usr/bin/env bash
#SBATCH --job-name=NUCLEI_SEGM
#SBATCH --output=NUCLEI_SEGM_%j.out
#SBATCH --error=NUCLEI_SEGM_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=124G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail
#--gres=gpu:1
#--partition=all -w izbdodoma

# ——— User parameters ———
#High res images - nuclei size needs to be changed in segment.py so far, be careful of memory requirements,
# might need to use --mem=124G and wait for resource allocation
RAW_DIR="/mnt/external.data/MeisterLab/mvolosko/image_project/high_res_images/1273/"
DENOISED_DIR="None"
MODEL_DIR="/mnt/external.data/MeisterLab/lhinder/segmentation_3d_anja/code/worms_1000epochs_v0"
OUT_ROOT="/mnt/external.data/MeisterLab/mvolosko/image_project/SDC1/1273_fast_imaging/"

# RAW_DIR="/mnt/external.data/MeisterLab/mvolosko/image_project/high_res_images/1268/"
# DENOISED_DIR="None"
# MODEL_DIR="/mnt/external.data/MeisterLab/lhinder/segmentation_3d_anja/code/worms_1000epochs_v0"
# OUT_ROOT="/mnt/external.data/MeisterLab/mvolosko/image_project/SDC1/1268_fast_imaging/"

#Dario Single images
# RAW_DIR='/mnt/external.data/MeisterLab/Dario/SDC1/1273/20240813_e/'
# MODEL_DIR="/mnt/external.data/MeisterLab/lhinder/segmentation_3d_anja/code/worms_1000epochs_v0"
# DENOISED_DIR=="${RAW_DIR}/N2V_dpy27_mSG_emr1_mCh/denoised/"
# OUT_ROOT="/mnt/external.data/MeisterLab/mvolosko/image_project/SDC1/1273/20240813_e/"

#Dario images heatshock timelapse
# RAW_DIR="/mnt/external.data/MeisterLab/Dario/SDC1/1273/20241108_e_hs"
# DENOISED_DIR="${RAW_DIR}/N2V_sdc1_dpy27_mSG_emr1_mCh/denoised"
# MODEL_DIR="/mnt/external.data/MeisterLab/lhinder/segmentation_3d_anja/code/worms_1000epochs_v0"
# OUT_ROOT="/mnt/external.data/MeisterLab/mvolosko/image_project/SDC1/1273/20241108_e_hs"

# RAW_DIR="/mnt/external.data/MeisterLab/Dario/SDC1/1273/20241010_e_tl/"
# DENOISED_DIR="${RAW_DIR}/N2V_sdc1_dpy27_mSG_emr1_mCh/denoised"
# MODEL_DIR="/mnt/external.data/MeisterLab/lhinder/segmentation_3d_anja/code/worms_1000epochs_v0"
# OUT_ROOT="/mnt/external.data/MeisterLab/mvolosko/image_project/SDC1/1273/20241010_e_tl/"

# RAW_DIR="/mnt/external.data/MeisterLab/Dario/DPY27/1268/20241107_e_hs/"
# DENOISED_DIR="${RAW_DIR}/N2V_sdc1_dpy27_mSG_emr1_mCh/denoised"
# MODEL_DIR="/mnt/external.data/MeisterLab/lhinder/segmentation_3d_anja/code/worms_1000epochs_v0"
# OUT_ROOT="/mnt/external.data/MeisterLab/mvolosko/image_project/DPY27/1268/20241107_e_hs/"

# RAW_DIR='/mnt/external.data/MeisterLab/Dario/DPY27/1268/20241010_e_tl/'
# DENOISED_DIR="${RAW_DIR}/N2V_sdc1_dpy27_mSG_emr1_mCh/denoised"
# MODEL_DIR="/mnt/external.data/MeisterLab/lhinder/segmentation_3d_anja/code/worms_1000epochs_v0"
# OUT_ROOT="/mnt/external.data/MeisterLab/mvolosko/image_project/DPY27/1268/20241010_e_tl/"

cd ../
# activate env
set +u
source /home/mvolosko/miniconda3/etc/profile.d/conda.sh
conda activate lhcellpose
set -u
# make sure output root exists
mkdir -p "${OUT_ROOT}"

# ——— 1. Segment & EDT (use raw as fallback if no denoised) ———
# python -m segmentation.cli segment \
#     --raw-dir      "${RAW_DIR}" \
#     --denoised-dir "${DENOISED_DIR}" \
#     --pattern      "*.nd2" \
#     --model        "${MODEL_DIR}" \
#     --out-root     "${OUT_ROOT}" \
#     --gpu \
#     --do-qc

# with no denoising step:
python -m segmentation.cli segment \
    --raw-dir  "${RAW_DIR}" \
    --pattern  "*.tif" \
    --model    "${MODEL_DIR}" \
    --out-root "${OUT_ROOT}" \
    --gpu \
    --do-qc \
    --blur-sigma "2.0"

# ——— 2. Distance–intensity analysis & summarization ———
python -m segmentation.cli analyze \
    --raw-dir  "${RAW_DIR}" \
    --seg-dir  "${OUT_ROOT}/segmentation" \
    --out-root "${OUT_ROOT}"

echo "✅ Pipeline completed – results in ${OUT_ROOT}"
