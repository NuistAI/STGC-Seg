#!/bin/bash
#SBATCH --job-name=sam3_breast
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=96:00:00
#SBATCH --gres=shard:1
#SBATCH --output=train_breast_log_%j.out
#SBATCH --error=train_breast_err_%j.out

source activate GXSAM310
export PYTHONUNBUFFERED=1

cd /home/Disk4toCheckpoint/gx/Medical-SAM3-main

GPU_ID=0
LABELED_RATIO=0.2
SEED=0
BATCH_SIZE=12
LR=0.0001
EPOCHS=200
CKPT=checkpoint/checkpoint.pt

echo "Starting BrEaST training at $(date)"
echo "GPU_ID=${GPU_ID} LABELED_RATIO=${LABELED_RATIO} SEED=${SEED} BATCH_SIZE=${BATCH_SIZE} LR=${LR} EPOCHS=${EPOCHS}"

python -u train_breast.py \
    --config train_breast.yaml \
    --gpu ${GPU_ID:-0} \
    --labeled_ratio ${LABELED_RATIO} \
    --seed ${SEED} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --epochs ${EPOCHS} \
    --checkpoint_path ${CKPT}

echo "Job finished at $(date)"
