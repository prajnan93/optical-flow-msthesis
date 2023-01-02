#!/bin/bash

#SBATCH --time=192:00:00
#SBATCH --job-name=nat_gm_kubric_baseline_steps_100k
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/transformers/outs/nat_gm_kubric_baseline_steps_100k.out

# GMFlowV2 training with baseline training settings and augmentations and GMFlow Normalization
# Neighborhood Attention Encoder

module load cuda/11.3
cd ../../
python train.py --model "GMFlowV2" \
                --model_cfg "./configs/transformers/models/nat_gm.yaml" \
                --train_cfg "./configs/transformers/trainer/kubric_baseline.yaml" \
                --device "0" \
                --log_dir "../results/transformers/logs/nat_gm_kubric_baseline_steps_100k" \
                --ckpt_dir "../results/transformers/ckpts/nat_gm_kubric_baseline_steps_100k" \
                --batch_size 10 \
                --start_iteration 1 \
                --num_steps 100100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496