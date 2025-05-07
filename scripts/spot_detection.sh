#!/bin/bash
#SBATCH --job-name=spot_analysis
#SBATCH --output=spot_analysis_%j.out
#SBATCH --error=spot_analysis_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1

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

RAW_DIR='/mnt/external.data/MeisterLab/Dario/SDC1/1273/20241010_e_tl/'
BASE_DIR="/mnt/external.data/MeisterLab/mvolosko/image_project/SDC1/1273/20241010_e_tl/"
INPUT_DIR="${RAW_DIR}/N2V_dpy27_mSG_emr1_mCh/denoised/"
NUCLEI_MASK_DIR="${BASE_DIR}/segmentation/"
NUCLEI_CSV_DIR="${BASE_DIR}/nuclei/"
OUTPUT_DIR="${BASE_DIR}/spots/"
###################################


# make sure the output directory exists
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
