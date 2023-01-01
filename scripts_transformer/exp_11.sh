#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --job-name=gmflow_exp11
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:8
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/gmflow/outs/exp11.out

# GMFlowV2 training with RAFT Augmentation, GMFflow Training settings(difference lies in loss fn gamma and scheduler anneal strategy) and GMFlow Normalization
# Tiny ViT Encoder
# Effective batch size = 8

module load cuda/11.3
cd ../
python train.py --model "GMFlowV2" \
                --model_cfg "./configs/gmflow/models/gmflow_v07.yaml" \
                --train_cfg "./configs/gmflow/trainer/chairs_v2_0.yaml" \
                --device "all" \
                --log_dir "../results/gmflow/logs/exp11" \
                --ckpt_dir "../results/gmflow/ckpts/exp11" \
                --batch_size 1 \
                --start_iteration 1 \
                --num_steps 400100 \
                --train_crop_size 384 512 \
                --val_crop_size 384 512 \
                --world_size 8