#!/bin/bash

#SBATCH --time=240:00:00
#SBATCH --job-name=pwcnet_exp002
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/pwcnet/outs/exp002.out

# PWCNet training with RAFT Augmentation, Training settings and PWCNet Normalization
# batch_size: 8

module load cuda/11.3
cd ../
python train.py --model "PWCNet" \
                --model_cfg "./configs/pwcnet/models/pwcnet.yaml" \
                --train_cfg "./configs/pwcnet/trainer/chairs_v1_2.yaml" \
                --device "0" \
                --log_dir "../results/pwcnet/logs/exp002" \
                --ckpt_dir "../results/pwcnet/ckpts/exp002" \
                --batch_size 8 \
                --start_iteration 1 \
                --num_steps 2400100 \
                --train_crop_size 384 448 \
                --val_crop_size 384 448 