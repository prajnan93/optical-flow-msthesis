#!/bin/bash

#SBATCH --time=240:00:00
#SBATCH --job-name=pwcnet_exp13
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/pwcnet/outs/exp13.out

# Pretrained PWCNet training with RAFT Augmentation, Training settings and PWCNet Normalization and Kubric dataset
# Pretrained on FlyingChairs
# batch_size: 8

module load cuda/11.3
cd ../
python train.py --model "PWCNet" \
                --model_cfg "./configs/pwcnet/models/pwcnet.yaml" \
                --train_cfg "./configs/pwcnet/trainer/kubric_v1_0.yaml" \
                --resume_ckpt "../results/pwcnet/ckpts/exp07/pwcnet_step1200000.pth"\
                --device "0" \
                --lr 0.000125 \
                --log_dir "../results/pwcnet/logs/exp13" \
                --ckpt_dir "../results/pwcnet/ckpts/exp13" \
                --batch_size 8 \
                --start_iteration 1 \
                --num_steps 1200100 \
                --train_crop_size 384 448 \
                --val_crop_size 384 448 