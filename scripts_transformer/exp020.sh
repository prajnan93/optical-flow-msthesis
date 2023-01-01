#!/bin/bash

#SBATCH --time=192:00:00
#SBATCH --job-name=gmflow_exp020
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:4
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/gmflow/outs/exp020.out

# GMFlowV2 training with RAFT Augmentation, GMFflow Training settings(difference lies in loss fn gamma and scheduler anneal strategy) and GMFlow Normalization
# Neighborhood Attention Transformer Encoder
# Effective batch size = 8

module load cuda/11.3
cd ../
python train.py --model "GMFlowV2" \
                --model_cfg "./configs/gmflow/models/gmflow_v13.yaml" \
                --train_cfg "./configs/gmflow/trainer/chairs_v2_0.yaml" \
                --device "all" \
                --log_dir "../results/gmflow/logs/exp020" \
                --ckpt_dir "../results/gmflow/ckpts/exp020" \
                --batch_size 2 \
                --start_iteration 1 \
                --num_steps 1200200 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496 \
                --world_size 4