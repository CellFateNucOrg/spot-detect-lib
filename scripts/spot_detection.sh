#!/bin/bash
#SBATCH --job-name=SPOT_det
#SBATCH --output=SPOT_det_%j.out
#SBATCH --error=SPOT_det_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --partition=all

##### set all your dirs here #####

# ——— User parameters ———
# RAW_DIR='/mnt/external.data/MeisterLab/Dario/DPY27/1268/20241010_e_tl/'
# BASE_DIR="/mnt/external.data/MeisterLab/mvolosko/image_project/DPY27/1268/20241010_e_tl/"
# INPUT_DIR="${RAW_DIR}/N2V_dpy27_mSG_emr1_mCh/denoised/"
# NUCLEI_MASK_DIR="${BASE_DIR}/segmentation/"
# NUCLEI_CSV_DIR="${BASE_DIR}/nuclei/"
# OUTPUT_DIR="${BASE_DIR}/spots/"

# RAW_DIR='/mnt/external.data/MeisterLab/Dario/SDC1/1273/20241108_e_hs/'
# BASE_DIR="/mnt/external.data/MeisterLab/mvolosko/image_project/SDC1/1273/20241108_e_hs/"
# INPUT_DIR="${RAW_DIR}/N2V_dpy27_mSG_emr1_mCh/denoised/"
# NUCLEI_MASK_DIR="${BASE_DIR}/segmentation/"
# NUCLEI_CSV_DIR="${BASE_DIR}/nuclei/"
# OUTPUT_DIR="${BASE_DIR}/spots/"

# RAW_DIR='/mnt/external.data/MeisterLab/Dario/DPY27/1268/20241107_e_hs/'
# BASE_DIR="/mnt/external.data/MeisterLab/mvolosko/image_project/DPY27/1268/20241107_e_hs/"
# INPUT_DIR="${RAW_DIR}/N2V_dpy27_mSG_emr1_mCh/denoised/"
# NUCLEI_MASK_DIR="${BASE_DIR}/segmentation/"
# NUCLEI_CSV_DIR="${BASE_DIR}/nuclei/"
# OUTPUT_DIR="${BASE_DIR}/spots/"

# RAW_DIR='/mnt/external.data/MeisterLab/Dario/SDC1/1273/20241010_e_tl/'
# BASE_DIR="/mnt/external.data/MeisterLab/mvolosko/image_project/SDC1/1273/20241010_e_tl/"
# INPUT_DIR="${RAW_DIR}/N2V_dpy27_mSG_emr1_mCh/denoised/"
# NUCLEI_MASK_DIR="${BASE_DIR}/segmentation/"
# NUCLEI_CSV_DIR="${BASE_DIR}/nuclei/"
# OUTPUT_DIR="${BASE_DIR}/spots/"

#Single Images
# RAW_DIR='/mnt/external.data/MeisterLab/Dario/SDC1/1273/20240813_e/'
# BASE_DIR="/mnt/external.data/MeisterLab/mvolosko/image_project/SDC1/1273/20240813_e/"
# INPUT_DIR="${RAW_DIR}/N2V_sdc1_dpy27_mSG_emr1_mCh/denoised/"
# NUCLEI_MASK_DIR="${BASE_DIR}/segmentation/"
# NUCLEI_CSV_DIR="${BASE_DIR}/nuclei/"
# OUTPUT_DIR="${BASE_DIR}/spots/"

# RAW_DIR='/mnt/external.data/MeisterLab/Dario/DPY27/1268/20240808_e/'
# BASE_DIR="/mnt/external.data/MeisterLab/mvolosko/image_project/dpy27/DPY27/1268/20240808_e/"
# INPUT_DIR="${RAW_DIR}/N2V_dpy27_mSG_emr1_mCh/denoised/"
# NUCLEI_MASK_DIR="${BASE_DIR}/segmentation/"
# NUCLEI_CSV_DIR="${BASE_DIR}/nuclei/"
# OUTPUT_DIR="${BASE_DIR}/spots/"

#High res images

# Test small subset of images 1268
RAW_DIR="/mnt/external.data/MeisterLab/mvolosko/image_project/high_res_images/1268/"
BASE_DIR="/mnt/external.data/MeisterLab/mvolosko/image_project/SDC1/test_subset/"
INPUT_DIR="${RAW_DIR}"
NUCLEI_MASK_DIR="${BASE_DIR}/segmentation/"
NUCLEI_CSV_DIR="${BASE_DIR}/nuclei/"
OUTPUT_DIR="${BASE_DIR}/spots/"

# Need the adjustments in the pixel size for this images, otherwise it takes too long and probably looks at wrong spots
# RAW_DIR="/mnt/external.data/MeisterLab/mvolosko/image_project/high_res_images/1268/"
# BASE_DIR="/mnt/external.data/MeisterLab/mvolosko/image_project/SDC1/1268_fast_imaging/"
# INPUT_DIR="${RAW_DIR}"
# NUCLEI_MASK_DIR="${BASE_DIR}/segmentation/"
# NUCLEI_CSV_DIR="${BASE_DIR}/nuclei/"
# OUTPUT_DIR="${BASE_DIR}/spots/"

# RAW_DIR="/mnt/external.data/MeisterLab/mvolosko/image_project/high_res_images/1273/"
# BASE_DIR="/mnt/external.data/MeisterLab/mvolosko/image_project/SDC1/1273_fast_imaging/"
# INPUT_DIR="${RAW_DIR}"
# NUCLEI_MASK_DIR="${BASE_DIR}/segmentation/"
# NUCLEI_CSV_DIR="${BASE_DIR}/nuclei/"
# OUTPUT_DIR="${BASE_DIR}/spots/"
##################################


# make sure the output directories exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$BASE_DIR"

# load your conda env
source /home/mvolosko/miniconda3/etc/profile.d/conda.sh
conda activate lhcellpose

# run the script
python -u ../spot_detection/main.py \
  --base-dir        "$BASE_DIR" \
  --input-dir       "$INPUT_DIR" \
  --nuclei-mask-dir "$NUCLEI_MASK_DIR" \
  --nuclei-csv-dir  "$NUCLEI_CSV_DIR" \
  --output-dir      "$OUTPUT_DIR" 
conda deactivate