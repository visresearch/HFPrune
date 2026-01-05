#!/bin/bash
#SBATCH --job-name=finetune      
#SBATCH --output=./log/lora/%j.txt           
#SBATCH --error=./log/lora/%j.txt           
#SBATCH --nodelist=node3
#SBATCH --partition=3090        
#SBATCH --nodes=1                 
#SBATCH --ntasks-per-node=1       
#SBATCH --gres=gpu:4              
#SBATCH --cpus-per-task=32

CONDA_BASE_PATH=""

source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"

CONDA_ENV_NAME="" 

conda activate "${CONDA_ENV_NAME}"

cd 

MODELS=(
)
LEARNING_RATES=(4e-4)


for model_path in "${STUDENT_MODELS[@]}"
do

    for lr_value in "${LEARNING_RATES[@]}"
    do
        accelerate launch --num_processes=4 --num_machines 1 \
        --mixed_precision bf16 --dynamo_backend no \
        fintune_lora.py \
        --gradient_accumulation_steps 32 \
        --model "${model_path}" \
        --lr "${lr_value}" \
        --dataset_name "out/cache/Lamini-llama3.2-clean-1024/" \
        --tokenizer_max 1024 \
        --lora_r 32 \
        --lora_alpha 64
    done
done


