#!/bin/bash

#SBATCH --time=96:00:00
#SBATCH --job-name=gmflow_exp02
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:4
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/gmflow/outs/exp02.out

# GMFlow training with RAFT Augmentation, GMFflow Training settings(difference lies in loss fn gamma and scheduler anneal strategy) and GMFlow Normalization
# Effective batch size = 16

module load cuda/11.3
cd ../
python train.py --model "GMFlow" \
                --model_cfg "./configs/gmflow/models/gmflow_v01.yaml" \
                --train_cfg "./configs/gmflow/trainer/chairs_v1_1.yaml" \
                --device "all" \
                --log_dir "../results/gmflow/logs/exp02" \
                --ckpt_dir "../results/gmflow/ckpts/exp02" \
                --batch_size 4 \
                --start_iteration 1 \
                --num_steps 100100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496 \
                --world_size 4