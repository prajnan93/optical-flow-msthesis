#!/bin/bash

#SBATCH --time=96:00:00
#SBATCH --job-name=gmflow_chairs_baseline
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/transformers/outs/gmflow_chairs_baseline.out

# GMFlow training with RAFT Augmentation, Training settings and GMFlow Normalization
# Effective batch size = 10

module load cuda/11.3
cd ../../
python train.py --model "GMFlow" \
                --model_cfg "./configs/transformers/models/gmflow.yaml" \
                --train_cfg "./configs/transformers/trainer/chairs_baseline.yaml" \
                --device "0" \
                --log_dir "../results/transformers/logs/gmflow_chairs_baseline" \
                --ckpt_dir "../results/transformers/ckpts/gmflow_chairs_baseline" \
                --batch_size 10 \
                --start_iteration 1 \
                --num_steps 100100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496