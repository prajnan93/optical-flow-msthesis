#!/bin/bash

#SBATCH --time=192:00:00
#SBATCH --job-name=swin_gm_hf_reduced_depth_kubrics_baseline_lr_125e_6
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:5
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/transformers/outs/swin_gm_hf_reduced_depth_kubrics_baseline_lr_125e_6.out

# GMFlowV2 training with baseline training settings and augmentations and GMFlow Normalization
# SwinV2 Encoder
# Lower learning rate : 0.000125
# Effective batch size = 10

module load cuda/11.3
cd ../../
python train.py --model "GMFlowV2" \
                --model_cfg "./configs/transformers/models/swin_huggingface_reduced_depth_gm.yaml" \
                --train_cfg "./configs/transformers/trainer/kubric_baseline_ddp.yaml" \
                --device "all" \
                --log_dir "../results/transformers/logs/swin_gm_hf_reduced_depth_kubrics_baseline_lr_125e_6" \
                --ckpt_dir "../results/transformers/ckpts/swin_gm_hf_reduced_depth_kubrics_baseline_lr_125e_6" \
                --lr 0.000125 \
                --batch_size 2 \
                --start_iteration 1 \
                --num_steps 1200100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496 \
                --world_size 5