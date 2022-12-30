#!/bin/bash

#SBATCH --time=240:00:00
#SBATCH --job-name=pwcnet_raft_encoder
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/pwcnet/outs/chairs_baseline_raft_encoder.out

# PWCNet with RAFT Encoder and baseline Augmentation
# batch_size: 8

module load cuda/11.3
cd ../
python train.py --model "PWCNet" \
                --model_cfg "./configs/pwcnet/models/pwcnet_raft_encoder.yaml" \
                --train_cfg "./configs/pwcnet/trainer/chairs_baseline.yaml" \
                --device "0" \
                --log_dir "../results/pwcnet/logs/chairs_baseline_raft_encoder" \
                --ckpt_dir "../results/pwcnet/ckpts/chairs_baseline_raft_encoder" \
                --batch_size 8 \
                --start_iteration 1 \
                --num_steps 1200100 \
                --train_crop_size 384 448 \
                --val_crop_size 384 448 