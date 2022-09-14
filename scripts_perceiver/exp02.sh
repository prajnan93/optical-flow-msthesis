#!/bin/bash

#SBATCH --time=240:00:00
#SBATCH --job-name=perceiver_exp02
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/perceiver/outs/exp02.out

# Perceiver IO training with perceiver Augmentation, Training settings and Normalization

module load cuda/11.3
cd ../
python train.py --model "Perceiver" \
                --model_cfg "./configs/perceiver/models/perceiver.yaml" \
                --train_cfg "./configs/perceiver/trainer/chairs_v2_0.yaml" \
                --device "0" \
                --log_dir "../results/perceiver/logs/exp02" \
                --ckpt_dir "../results/perceiver/ckpts/exp02" \
                --batch_size 10 \
                --start_iteration 1 \
                --num_steps 100100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496 