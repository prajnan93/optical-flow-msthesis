#!/bin/bash

#SBATCH --time=192:00:00
#SBATCH --job-name=pwcnet_kubric_baseline_lr_disrupt_1200k_steps
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/pwcnet/outs/kubric_baseline_lr_disrupt_1200k_steps.out

# Resume PWCNet training with RAFT Augmentation, Training settings and PWCNet Normalization
# following Kubric -> Kubric schedule with a lower learning rate of 0.000125
# batch_size: 8

module load cuda/11.3
cd ../
python train.py --model "PWCNet" \
                --model_cfg "./configs/pwcnet/models/pwcnet.yaml" \
                --train_cfg "./configs/pwcnet/trainer/kubric_baseline_lr_disrupt.yaml" \
                --device "0" \
                --log_dir "../results/pwcnet/logs/kubric_baseline_lr_disrupt_1200k_steps" \
                --ckpt_dir "../results/pwcnet/ckpts/kubric_baseline_lr_disrupt_1200k_steps" \
                --resume_ckpt "../results/pwcnet/ckpts/kubric_baseline_1200k_steps/pwcnet_step1200000.pth" \
                --lr 0.000125 \
                --batch_size 8 \
                --start_iteration 1 \
                --num_steps 1200100 \
                --train_crop_size 384 448 \
                --val_crop_size 384 448 \
                --freeze_batch_norm