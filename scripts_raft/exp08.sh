#!/bin/bash

#SBATCH --time=96:00:00
#SBATCH --job-name=raft_exp08
#SBATCH --partition=jiang
#SBATCH --mem=48G
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/raft/outs/exp08.out

# RAFT training with Autoflow Augmentation, Training settings and Normalization and Kubric dataset
# noise aug_prob: 1
# out of boundary cropping: True 

module load cuda/11.3
cd ../
python train.py --model "RAFT" \
                --model_cfg "./configs/raft/models/raft.yaml" \
                --train_cfg "./configs/raft/trainer/kubric_v1_2.yaml" \
                --device "0" \
                --log_dir "../results/raft/logs/exp08" \
                --ckpt_dir "../results/raft/ckpts/exp08" \
                --batch_size 10 \
                --start_iteration 1 \
                --num_steps 100100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496 