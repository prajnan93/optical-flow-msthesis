#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --job-name=gmflow_exp07
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:4
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/gmflow/outs/exp07.out

# GMFlowV2 training with RAFT Augmentation, GMFflow Training settings(difference lies in loss fn gamma and scheduler anneal strategy) and GMFlow Normalization
# SwinV2 Encoder
# Effective batch size = 16

module load cuda/11.3
cd ../
python train.py --model "GMFlowV2" \
                --model_cfg "./configs/gmflow/models/gmflow_v04.yaml" \
                --train_cfg "./configs/gmflow/trainer/chairs_v2_0.yaml" \
                --device "all" \
                --log_dir "../results/gmflow/logs/exp07" \
                --ckpt_dir "../results/gmflow/ckpts/exp07" \
                --batch_size 4 \
                --start_iteration 1 \
                --num_steps 400100 \
                --train_crop_size 384 512 \
                --val_crop_size 384 512 \
                --world_size 4