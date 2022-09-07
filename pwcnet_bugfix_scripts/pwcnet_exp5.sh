#!/bin/bash

#SBATCH --time=120:00:00
#SBATCH --job-name=pwcnet_bugfix_exp5
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/pwcnet_bugfix/outs/exp5.out

module load cuda/11.3
cd ../
python train.py --model "PWCNetV2" \
                --model_cfg "./configs/pwcnet_bugfix/models/nnflow_v3.yaml" \
                --train_cfg "./configs/pwcnet_bugfix/trainer/chairs_v4_1.yaml" \
                --device "0" \
                --log_dir "../results/pwcnet_bugfix/logs/exp5" \
                --ckpt_dir "../results/pwcnet_bugfix/ckpts/exp5" \
                --batch_size 8 \
                --epochs 432 \
                --train_crop_size 384 448 \
                --val_crop_size 384 448 