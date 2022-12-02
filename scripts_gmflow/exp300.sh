#!/bin/bash

#SBATCH --time=192:00:00
#SBATCH --job-name=gmflow_raft_encoder_exp300
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/gmflow/outs/exp300.out

# GMFlowV2 training with RAFT Augmentation, GMFflow Training settings(difference lies in loss fn gamma and scheduler anneal strategy) and GMFlow Normalization
# RESIDUAL ENCODER with normalization

module load cuda/11.3
cd ../
python train.py --model "GMFlowV2" \
                --model_cfg "./configs/gmflow/models/gmflow_v14.yaml" \
                --train_cfg "./configs/gmflow/trainer/chairs_v2_1.yaml" \
                --device "0" \
                --log_dir "../results/gmflow/logs/exp300" \
                --ckpt_dir "../results/gmflow/ckpts/exp300" \
                --batch_size 10 \
                --start_iteration 1 \
                --num_steps 100100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496