#!/bin/bash

#SBATCH --time=08:00:00
#SBATCH --job-name=pwcnet_bugfix_exp14
#SBATCH --partition=gpu
#SBATCH --mem=24G
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/pwcnet_bugfix/outs/exp14.out

# Ezflow PWCNet Refactor experiment

module load cuda/11.3
cd ../
python train.py --model "PWCNetV3" \
                --model_cfg "./configs/pwcnet_bugfix/models/ezflow.yaml" \
                --train_cfg "./configs/pwcnet_bugfix/trainer/chairs_v5_2.yaml" \
                --device "0" \
                --log_dir "../results/pwcnet_bugfix/logs/exp14" \
                --ckpt_dir "../results/pwcnet_bugfix/ckpts/exp14" \
                --batch_size 8 \
                --start_iteration 1 \
                --num_steps 100100 \
                --train_crop_size 384 448 \
                --val_crop_size 384 448 