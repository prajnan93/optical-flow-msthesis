#!/bin/bash

#SBATCH --time=96:00:00
#SBATCH --job-name=raft_exp07
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/raft/outs/exp07.out

# RAFT training with Autoflow Augmentation, Training settings and Normalization and Kubric dataset
# noise aug_prob: 1
# out of boundary cropping: False 

module load cuda/11.3
cd ../
python train.py --model "RAFT" \
                --model_cfg "./configs/raft/models/raft.yaml" \
                --train_cfg "./configs/raft/trainer/kubric_v1_1.yaml" \
                --device "0" \
                --log_dir "../results/raft/logs/exp07" \
                --ckpt_dir "../results/raft/ckpts/exp07" \
                --batch_size 10 \
                --start_iteration 1 \
                --num_steps 100100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496 