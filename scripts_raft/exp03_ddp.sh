#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --job-name=raft_exp03_ddp
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:2
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/raft/outs/exp03_ddp.out

# RAFT training with RAFT Augmentation, Training settings and raft Normalization
# Effective batch size = 10

# moved Sync Batch Norm before wrapping model with DDP in EzFlow
# added distributed barrier to sync all process

module load cuda/11.3
cd ../
python train.py --model "RAFT" \
                --model_cfg "./configs/raft/models/raft.yaml" \
                --train_cfg "./configs/raft/trainer/chairs_v1_0_ddp.yaml" \
                --device "all" \
                --log_dir "../results/raft/logs/exp03_ddp" \
                --ckpt_dir "../results/raft/ckpts/exp03_ddp" \
                --lr 0.0008 \
                --batch_size 5 \
                --start_iteration 1 \
                --num_steps 100100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496 \
                --world_size 2