#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --job-name=raft_exp204
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/raft/outs/exp204.out

# RAFT training with RAFT Augmentation, Training settings and Normalization and Kubric dataset
# min scale test with same experiment as exp200

module load cuda/11.3
cd ../
python train.py --model "RAFT" \
                --model_cfg "./configs/raft/models/raft.yaml" \
                --train_cfg "./configs/raft/trainer/kubric_v1_0.yaml" \
                --device "0" \
                --log_dir "../results/raft/logs/exp204" \
                --ckpt_dir "../results/raft/ckpts/exp204" \
                --ckpt_interval 10000 \
                --batch_size 10 \
                --start_iteration 1 \
                --num_steps 100100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496 