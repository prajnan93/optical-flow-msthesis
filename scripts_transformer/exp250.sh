#!/bin/bash

#SBATCH --time=192:00:00
#SBATCH --job-name=gmflow_exp250
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/gmflow/outs/exp250.out

# GMFlowV2 training with RAFT Augmentation, GMFflow Training settings(difference lies in loss fn gamma and scheduler anneal strategy) and GMFlow Normalization
# Neighborhood Attention Encoder

module load cuda/11.3
cd ../
python train.py --model "GMFlowV2" \
                --model_cfg "./configs/gmflow/models/gmflow_v13.yaml" \
                --train_cfg "./configs/gmflow/trainer/kubrics_v1_3.yaml" \
                --device "0" \
                --log_dir "../results/gmflow/logs/exp250" \
                --ckpt_dir "../results/gmflow/ckpts/exp250" \
                --batch_size 10 \
                --start_iteration 1 \
                --num_steps 100100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496