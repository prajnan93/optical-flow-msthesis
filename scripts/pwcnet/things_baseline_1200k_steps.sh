#!/bin/bash

#SBATCH --time=240:00:00
#SBATCH --job-name=pwcnet_things_baseline_1200k_steps
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/pwcnet/outs/things_baseline_1200k_steps.out

# Resume PWCNet training with RAFT Augmentation, Training settings and PWCNet Normalization following Chairs -> Things schedule
# batch_size: 4

module load cuda/11.3
cd ../../
python train.py --model "PWCNet" \
                --model_cfg "./configs/pwcnet/models/pwcnet.yaml" \
                --train_cfg "./configs/pwcnet/trainer/things_baseline.yaml" \
                --device "0" \
                --log_dir "../results/pwcnet/logs/things_baseline_1200k_steps" \
                --ckpt_dir "../results/pwcnet/ckpts/things_baseline_1200k_steps" \
                --resume_ckpt "../results/pwcnet/ckpts/chairs_baseline_1200k_steps/pwcnet_step1200000.pth" \
                --batch_size 4 \
                --start_iteration 1 \
                --num_steps 1200100 \
                --train_crop_size 384 768 \
                --val_crop_size 384 448