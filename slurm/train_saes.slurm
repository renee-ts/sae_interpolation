#!/bin/bash
#SBATCH --job-name=train-saes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4        
#SBATCH --mem=32G                
#SBATCH --time=6:00:00           
#SBATCH --partition=a40       
#SBATCH --qos=normal             
#SBATCH --gres=gpu:1             
#SBATCH --array=0-3%4            
#SBATCH --output=logs/%x_%A_%a.out

source ~/.bashrc
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Python version: $(python --version)"

TOTAL_TASKS=4
TASK_ID=${SLURM_ARRAY_TASK_ID}

python3 main.py \
  --config config/config.yaml \
  --output_dir /checkpoint/$USER/$SLURM_JOB_ID \
  --total_tasks 4 \
  --task_id ${SLURM_ARRAY_TASK_ID}