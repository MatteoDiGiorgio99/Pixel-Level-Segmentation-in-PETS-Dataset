#!/bin/bash

#SBATCH --job-name=OxfordPetDatasetMainOptimizer
#SBATCH --output=%x.o
#SBATCH --error=%x.e
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:2
#SBATCH --mem=32gb
#SBATCH --time=0-24:00:00
##SBATCH --account=matteo.digiorgio

module load miniconda3
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate myenv

python main[Optimizer].py  # Change this line to the name of the script you want to run

conda deactivate
