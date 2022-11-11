#!/bin/bash

#SBATCH --time=192:00:00
#SBATCH --job-name=pwcnet_exp201
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/pwcnet/outs/exp201.out

# Resume PWCNet training with RAFT Augmentation, Training settings and PWCNet Normalization following Chairs -> Things schedule
# batch_size: 8

module load cuda/11.3
cd ../
python train.py --model "PWCNet" \
                --model_cfg "./configs/pwcnet/models/pwcnet.yaml" \
                --train_cfg "./configs/pwcnet/trainer/kubric_v1_1.yaml" \
                --device "0" \
                --log_dir "../results/pwcnet/logs/exp201" \
                --ckpt_dir "../results/pwcnet/ckpts/exp201" \
                --resume_ckpt "../results/pwcnet/ckpts/exp200/pwcnet_step1200000.pth" \
                --batch_size 8 \
                --start_iteration 1 \
                --num_steps 1200100 \
                --train_crop_size 384 448 \
                --val_crop_size 384 448 \
                --freeze_batch_norm