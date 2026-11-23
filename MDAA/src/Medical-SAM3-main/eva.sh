#!/bin/bash
#SBATCH --job-name=sam3_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --gres=shard:1
#SBATCH --output=train_log_%j.out
#SBATCH --error=train_err_%j.out

source activate GXSAM310
export PYTHONUNBUFFERED=1

echo "Starting training job at $(date)"
echo "Using Node: $SLURM_NODELIST"

python -u infer_cam.py

echo "Job finished at $(date)"