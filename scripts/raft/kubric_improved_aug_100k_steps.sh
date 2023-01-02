#!/bin/bash

#SBATCH --time=32:00:00
#SBATCH --job-name=raft_kubric_improved_aug_100k_steps
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/raft/outs/kubric_improved_aug_100k_steps.out

# RAFT training with RAFT Training settings, Kubric dataset and AutoFlow Augmentations

module load cuda/11.3
cd ../../
python train.py --model "RAFT" \
                --model_cfg "./configs/raft/models/raft.yaml" \
                --train_cfg "./configs/raft/trainer/kubric_improved_aug.yaml" \
                --device "0" \
                --log_dir "../results/raft/logs/kubric_improved_aug_100k_steps" \
                --ckpt_dir "../results/raft/ckpts/kubric_improved_aug_100k_steps" \
                --ckpt_interval 20000 \
                --batch_size 10 \
                --start_iteration 1 \
                --num_steps 100100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496 