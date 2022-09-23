#!/bin/bash

#SBATCH --time=240:00:00
#SBATCH --job-name=pwcnet_exp09
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/pwcnet/outs/exp09.out

# PWCNet training with Autoflow Augmentation, RAFT Training settings and PWCNet Normalization
# noise aug_prob:1
# out of boundary cropping: True
# batch_size: 8

module load cuda/11.3
cd ../
python train.py --model "PWCNet" \
                --model_cfg "./configs/pwcnet/models/pwcnet.yaml" \
                --train_cfg "./configs/pwcnet/trainer/chairs_v2_4.yaml" \
                --device "0" \
                --log_dir "../results/pwcnet/logs/exp09" \
                --ckpt_dir "../results/pwcnet/ckpts/exp09" \
                --batch_size 8 \
                --start_iteration 1 \
                --num_steps 1200100 \
                --train_crop_size 384 448 \
                --val_crop_size 384 448 