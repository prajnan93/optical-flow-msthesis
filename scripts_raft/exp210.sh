#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --job-name=raft_kubric_longer_training_improved_augmentation
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/raft/outs/exp210.out

# RAFT training with RAFT Training settings, Kubric dataset and AutoFlow Augmentations
# Disable out-of-boundary crop
# longer training with improved augmentations

module load cuda/11.3
cd ../
python train.py --model "RAFT" \
                --model_cfg "./configs/raft/models/raft.yaml" \
                --train_cfg "./configs/raft/trainer/kubric_v2_5.yaml" \
                --device "0" \
                --log_dir "../results/raft/logs/exp210" \
                --ckpt_dir "../results/raft/ckpts/exp210" \
                --ckpt_interval 10000 \
                --batch_size 10 \
                --start_iteration 1 \
                --num_steps 200100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496 