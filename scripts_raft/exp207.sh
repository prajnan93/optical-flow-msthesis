#!/bin/bash

#SBATCH --time=32:00:00
#SBATCH --job-name=raft_exp207
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/raft/outs/exp207.out

# RAFT training with RAFT Training settings, Kubric dataset and AutoFlow Augmentations
# Perform Hard augmentation for Random noise

module load cuda/11.3
cd ../
python train.py --model "RAFT" \
                --model_cfg "./configs/raft/models/raft.yaml" \
                --train_cfg "./configs/raft/trainer/kubric_v2_4.yaml" \
                --device "0" \
                --log_dir "../results/raft/logs/exp207" \
                --ckpt_dir "../results/raft/ckpts/exp207" \
                --ckpt_interval 20000 \
                --batch_size 10 \
                --start_iteration 1 \
                --num_steps 100100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496 