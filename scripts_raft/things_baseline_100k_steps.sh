#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --job-name=raft_things_baseline_100k_steps
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/raft/outs/things_baseline_100k_steps.out

# Resume RAFT training with RAFT Augmentation, Training settings and Normalization following Chairs -> Things schedule

module load cuda/11.3
cd ../
python train.py --model "RAFT" \
                --model_cfg "./configs/raft/models/raft.yaml" \
                --train_cfg "./configs/raft/trainer/things_baseline.yaml" \
                --device "0" \
                --log_dir "../results/raft/logs/things_baseline_100k_steps" \
                --ckpt_dir "../results/raft/ckpts/things_baseline_100k_steps" \
                --resume_ckpt "../results/raft/ckpts/chairs_baseline_100k_steps/raft_step100000.pth" \
                --batch_size 6 \
                --start_iteration 1 \
                --num_steps 100100 \
                --train_crop_size 400 720 \
                --val_crop_size 368 496 \
                --freeze_batch_norm