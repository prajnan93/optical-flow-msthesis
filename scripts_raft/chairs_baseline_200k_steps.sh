#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --job-name=raft_chairs_baseline_200k_steps
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/raft/outs/chairs_baseline_200k_steps.out

# raft training with RAFT Augmentation, Training settings and Normalization

module load cuda/11.3
cd ../
python train.py --model "RAFT" \
                --model_cfg "./configs/raft/models/raft.yaml" \
                --train_cfg "./configs/raft/trainer/chairs_baseline.yaml" \
                --device "0" \
                --log_dir "../results/raft/logs/chairs_baseline_200k_steps" \
                --ckpt_dir "../results/raft/ckpts/chairs_baseline_200k_steps" \
                --batch_size 10 \
                --start_iteration 1 \
                --num_steps 200100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496 