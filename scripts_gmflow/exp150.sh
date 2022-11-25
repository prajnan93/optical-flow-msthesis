#!/bin/bash

#SBATCH --time=192:00:00
#SBATCH --job-name=gmflow_exp150
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:4
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/gmflow/outs/exp150.out

# GMFlowV2 training with RAFT Augmentation, GMFflow Training settings(difference lies in loss fn gamma and scheduler anneal strategy) and GMFlow Normalization
# Chairs -> Things schedule
# Neighborhood Attention Transformer Encoder
# Effective batch size = 8

module load cuda/11.3
cd ../
python train.py --model "GMFlowV2" \
                --model_cfg "./configs/gmflow/models/gmflow_v13.yaml" \
                --train_cfg "./configs/gmflow/trainer/things_v1_1.yaml" \
                --device "0" \
                --log_dir "../results/gmflow/logs/exp150" \
                --ckpt_dir "../results/gmflow/ckpts/exp150" \
                --resume_ckpt "../results/gmflow/ckpts/exp050/gmflowv2_step100000.pth" \
                --batch_size 10 \
                --start_iteration 1 \
                --num_steps 100100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496 \
                --freeze_batch_norm