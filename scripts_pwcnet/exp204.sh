#!/bin/bash

#SBATCH --time=192:00:00
#SBATCH --job-name=pwcnet_exp204
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/pwcnet/outs/exp204.out

# PWCNet training with RAFT Training settings and AutoFlow augmentations
# Kubric dataset
# Perform hard augmentation for Asymmetric Color Probability
# batch_size: 8

module load cuda/11.3
cd ../
python train.py --model "PWCNet" \
                --model_cfg "./configs/pwcnet/models/pwcnet.yaml" \
                --train_cfg "./configs/pwcnet/trainer/kubric_v2_1.yaml" \
                --device "0" \
                --log_dir "../results/pwcnet/logs/exp204" \
                --ckpt_dir "../results/pwcnet/ckpts/exp204" \
                --batch_size 8 \
                --start_iteration 1 \
                --num_steps 1200100 \
                --train_crop_size 384 448 \
                --val_crop_size 384 448 