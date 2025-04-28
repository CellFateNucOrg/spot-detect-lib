#!/usr/bin/env bash
#SBATCH --job-name=NUCLEI_SEGM
#SBATCH --output=NUCLEI_SEGM_%j.out
#SBATCH --error=NUCLEI_SEGM_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail

# ——— User parameters ———
# RAW_DIR="/mnt/external.data/MeisterLab/Dario/SDC1/1273/20241108_e_hs"
# DENOISED_DIR="${RAW_DIR}/N2V_sdc1_dpy27_mSG_emr1_mCh/denoised"
# MODEL_DIR="/mnt/external.data/MeisterLab/lhinder/segmentation_3d_anja/code/worms_1000epochs_v0"
# OUT_ROOT="/mnt/external.data/MeisterLab/mvolosko/image_project/SDC1/1273/20241108_e_hs"

RAW_DIR='/mnt/external.data/MeisterLab/Dario/DPY27/1268/20241010_e_tl/'
DENOISED_DIR="${RAW_DIR}/N2V_sdc1_dpy27_mSG_emr1_mCh/denoised"
MODEL_DIR="/mnt/external.data/MeisterLab/lhinder/segmentation_3d_anja/code/worms_1000epochs_v0"
OUT_ROOT="/mnt/external.data/MeisterLab/mvolosko/image_project/DPY27/1268/20241010_e_tl/"
cd ../
# activate env
set +u
source /home/mvolosko/miniconda3/etc/profile.d/conda.sh
conda activate lhcellpose
set -u
# make sure output root exists
mkdir -p "${OUT_ROOT}"

# # ——— 1. Segment & EDT (use raw as fallback if no denoised) ———
# python -m segmentation.cli segment \
#     --raw-dir      "${RAW_DIR}" \
#     --denoised-dir "${DENOISED_DIR}" \
#     --pattern      "*.nd2" \
#     --model        "${MODEL_DIR}" \
#     --out-root     "${OUT_ROOT}" \
#     --gpu \
#     --do-qc

# # if you also have .czi images with no denoising step:
# python -m segmentation.cli segment \
#     --raw-dir  "${RAW_DIR}" \
#     --pattern  "*.czi" \
#     --model    "${MODEL_DIR}" \
#     --out-root "${OUT_ROOT}" \
#     --gpu

# ——— 2. Distance–intensity analysis & summarization ———
python -m segmentation.cli analyze \
    --raw-dir  "${RAW_DIR}" \
    --seg-dir  "${OUT_ROOT}/segmentation" \
    --out-root "${OUT_ROOT}"

echo "✅ Pipeline complete – results in ${OUT_ROOT}"
