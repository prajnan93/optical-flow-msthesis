#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --job-name=swin_gm_chairs_baseline
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:5
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/transformers/outs/swin_gm_chairs_baseline.out

# GMFlowV2 training with baseline training settings and augmentations and GMFlow Normalization
# Effective batch size = 10

module load cuda/11.3
cd ../../
python train.py --model "GMFlowV2" \
                --model_cfg "./configs/transformers/models/swin_gm.yaml" \
                --train_cfg "./configs/transformers/trainer/chairs_baseline_ddp.yaml" \
                --device "all" \
                --log_dir "../results/transformers/logs/swin_gm_chairs_baseline" \
                --ckpt_dir "../results/transformers/ckpts/swin_gm_chairs_baseline" \
                --batch_size 2 \
                --start_iteration 1 \
                --num_steps 200100 \
                --train_crop_size 384 512 \
                --val_crop_size 384 512 \
                --world_size 5