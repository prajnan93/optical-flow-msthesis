#!/bin/bash

#SBATCH --time=192:00:00
#SBATCH --job-name=nat_gm_things_baseline_steps_100k
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/transformers/outs/nat_gm_things_baseline_steps_100k.out

# GMFlowV2 training with baseline training settings and augmentations and GMFlow Normalization
# Chairs -> Things schedule
# Neighborhood Attention Transformer Encoder
# Effective batch size = 8

module load cuda/11.3
cd ../
python train.py --model "GMFlowV2" \
                --model_cfg "./configs/transformers/models/nat_gm.yaml" \
                --train_cfg "./configs/transformers/trainer/things_baseline.yaml" \
                --device "0" \
                --log_dir "../results/transformers/logs/nat_gm_things_baseline_steps_100k" \
                --ckpt_dir "../results/transformers/ckpts/nat_gm_things_baseline_steps_100k" \
                --resume_ckpt "../results/transformers/ckpts/nat_gm_chairs_baseline_steps_100k/gmflowv2_step100000.pth" \
                --batch_size 6 \
                --start_iteration 1 \
                --num_steps 100100 \
                --train_crop_size 400 720 \
                --val_crop_size 400 720