#!/bin/bash

#SBATCH --time=192:00:00
#SBATCH --job-name=gmflow_no_norm_chairs_baseline
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/transformers/outs/gmflow_no_norm_chairs_baseline.out

# GMFlowV2 training with baseline training settings and augmentations and GMFlow Normalization
# RESIDUAL ENCODER without normalization

module load cuda/11.3
cd ../../
python train.py --model "GMFlowV2" \
                --model_cfg "./configs/transformers/models/gmflow_no_norm.yaml" \
                --train_cfg "./configs/transformers/trainer/chairs_baseline.yaml" \
                --device "0" \
                --log_dir "../results/transformers/logs/gmflow_no_norm_chairs_baseline" \
                --ckpt_dir "../results/transformers/ckpts/gmflow_no_norm_chairs_baseline" \
                --batch_size 10 \
                --start_iteration 1 \
                --num_steps 100100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496