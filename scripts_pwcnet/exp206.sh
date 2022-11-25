#!/bin/bash

#SBATCH --time=192:00:00
#SBATCH --job-name=pwcnet_exp206
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/pwcnet/outs/exp206.out

# PWCNet training with RAFT Training settings and AutoFlow augmentations
# Kubric dataset
# Disable random noise
# batch_size: 8

module load cuda/11.3
cd ../
python train.py --model "PWCNet" \
                --model_cfg "./configs/pwcnet/models/pwcnet.yaml" \
                --train_cfg "./configs/pwcnet/trainer/kubric_v2_3.yaml" \
                --device "0" \
                --log_dir "../results/pwcnet/logs/exp206" \
                --ckpt_dir "../results/pwcnet/ckpts/exp206" \
                --batch_size 8 \
                --start_iteration 1 \
                --num_steps 1200100 \
                --train_crop_size 384 448 \
                --val_crop_size 384 448 