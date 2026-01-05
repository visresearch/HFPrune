#!/bin/bash
#SBATCH --job-name=prune      
#SBATCH --output=./log/prune/%j.txt           
#SBATCH --error=./log/prune/%j.txt            
#SBATCH --nodelist=node3
#SBATCH --partition=3090         
#SBATCH --nodes=1                 
#SBATCH --ntasks-per-node=1       
#SBATCH --gres=gpu:4  
#SBATCH --cpus-per-task=32
# 输出一些作业信息

CONDA_BASE_PATH=""

source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
CONDA_ENV_NAME="" # 
echo "${CONDA_ENV_NAME}"
conda activate "${CONDA_ENV_NAME}"

python prune.py --seed 42 --mlp_ratio 2.8 --origin_path ""

