#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --job-name=raft_exp01_ddp
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/raft/outs/exp01_ddp.out

# RAFT training with RAFT Augmentation, Training settings and raft Normalization
# Effective batch size = 10

module load cuda/11.3
cd ../
python train.py --model "raft" \
                --model_cfg "./configs/raft/models/raft.yaml" \
                --train_cfg "./configs/raft/trainer/chairs_v1_0.yaml" \
                --device "all" \
                --log_dir "../results/raft/logs/exp01_ddp" \
                --ckpt_dir "../results/raft/ckpts/exp01_ddp" \
                --batch_size 5 \
                --start_iteration 1 \
                --num_steps 100100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496 \
                --world_size 2