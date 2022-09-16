#!/bin/bash

#SBATCH --time=240:00:00
#SBATCH --job-name=perceiver_exp01
#SBATCH --partition=jiang
#SBATCH --mem=96G
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/perceiver/outs/exp01.out

# Perceiver IO training with perceiver Augmentation, Training settings and Normalization

module load cuda/11.3
cd ../
python train.py --model "Perceiver" \
                --model_cfg "./configs/perceiver/models/perceiver.yaml" \
                --train_cfg "./configs/perceiver/trainer/chairs_ddp_v1_0.yaml" \
                --device "all" \
                --log_dir "../results/perceiver/logs/exp01" \
                --ckpt_dir "../results/perceiver/ckpts/exp01" \
                --batch_size 4 \
                --start_iteration 1 \
                --num_steps 100100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496 \
                --world_size 2