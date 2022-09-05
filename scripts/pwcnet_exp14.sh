#!/bin/bash

#SBATCH --time=08:00:00
#SBATCH --job-name=pwcnet_exp14
#SBATCH --partition=gpu
#SBATCH --mem=24G
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/pwcnet/outs/exp14.out

# Ezflow PWCNet Refactor experiment

module load cuda/11.3
cd ../
python train.py --model "PWCNetV3" \
                --model_cfg "./configs/pwcnet/models/ezflow.yaml" \
                --train_cfg "./configs/pwcnet/trainer/chairs_v4_5.yaml" \
                --device "0" \
                --log_dir "../results/pwcnet/logs/exp14" \
                --ckpt_dir "../results/pwcnet/ckpts/exp14" \
                --batch_size 8 \
                --num_steps 100000 \
                --train_crop_size 384 448 \
                --val_crop_size 384 448 