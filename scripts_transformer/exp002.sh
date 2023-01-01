#!/bin/bash

#SBATCH --time=192:00:00
#SBATCH --job-name=gmflow_exp002
#SBATCH --partition=jiang
#SBATCH --mem=96G
#SBATCH --gres=gpu:a5000:4
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/gmflow/outs/exp002.out

# GMFlowV2 training with RAFT Augmentation, GMFflow Training settings(difference lies in loss fn gamma and scheduler anneal strategy) and GMFlow Normalization
# SwinV2 Encoder
# Effective batch size = 16

module load cuda/11.3
cd ../
python train.py --model "GMFlowV2" \
                --model_cfg "./configs/gmflow/models/gmflow_v10.yaml" \
                --train_cfg "./configs/gmflow/trainer/chairs_v2_0.yaml" \
                --device "all" \
                --log_dir "../results/gmflow/logs/exp002" \
                --ckpt_dir "../results/gmflow/ckpts/exp002" \
                --batch_size 4 \
                --start_iteration 1 \
                --num_steps 1200100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496 \
                --world_size 4