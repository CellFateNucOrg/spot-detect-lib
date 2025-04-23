# spot-detect-lib

This repository focuses on spot detection in microscopic images with segmened nuclei.

The segmentation is adopted from the previous script and so far is located in Jupyter Notebook "5d_nuclei_segmentation...". It's slightly buggy because of old libraries used there and is to be integrated in the main modulized pipeline.

After one obtain nuclei segmentation data. The further nuclei detection is done with the script **main.py**. It could be also used as a Slurm script located in the folder **scripts_sbatch**. It authomatically activates the conda environment from my folder. 

The follow-up QC could involve 3d reconstruction of the spots within the nuclei. To do so you need to run a script in **utils** folder called **visualization.py** after changing the variables inside. And update for randomized qc is to be done.

