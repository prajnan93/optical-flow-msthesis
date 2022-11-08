#!/bin/bash

#SBATCH --time=32:00:00
#SBATCH --job-name=gmflow_exp09
#SBATCH --partition=jiang
#SBATCH --mem=384G
#SBATCH --gres=gpu:a6000:8
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/gmflow/outs/exp09.out

# GMFlow training with RAFT Augmentation, GMFflow Training settings(difference lies in loss fn gamma and scheduler anneal strategy) and GMFlow Normalization
# Pretrained Dino ViT Encoder FREEZED
# Effective batch size = 8

module load cuda/11.3
cd ../
python train.py --model "GMFlowV2" \
                --model_cfg "./configs/gmflow/models/gmflow_v05.yaml" \
                --train_cfg "./configs/gmflow/trainer/chairs_v2_0.yaml" \
                --device "all" \
                --log_dir "../results/gmflow/logs/exp09" \
                --ckpt_dir "../results/gmflow/ckpts/exp09" \
                --batch_size 1 \
                --start_iteration 1 \
                --num_steps 400100 \
                --train_crop_size 384 512 \
                --val_crop_size 384 512 \
                --world_size 8