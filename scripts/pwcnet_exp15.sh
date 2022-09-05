#!/bin/bash

#SBATCH --time=240:00:00
#SBATCH --job-name=pwcnet_exp15
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/pwcnet/outs/exp15.out

# Ezflow PWCNet Refactor experiment

module load cuda/11.3
cd ../
python train.py --model "PWCNetV3" \
                --model_cfg "./configs/pwcnet/models/ezflow.yaml" \
                --train_cfg "./configs/pwcnet/trainer/chairs_v4_5.yaml" \
                --device "0" \
                --log_dir "../results/pwcnet/logs/exp15" \
                --ckpt_dir "../results/pwcnet/ckpts/exp15" \
                --batch_size 8 \
                --num_steps 100000 \
                --train_crop_size 384 448 \
                --val_crop_size 384 448 