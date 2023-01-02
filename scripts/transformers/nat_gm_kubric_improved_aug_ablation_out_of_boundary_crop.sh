#!/bin/bash

#SBATCH --time=192:00:00
#SBATCH --job-name=nat_gm_kubric_improved_aug_ablation_out_of_boundary_crop
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/transformers/outs/nat_gm_kubric_improved_aug_ablation_out_of_boundary_crop.out

# GMFlowV2 training with baseline training settings, autoflow augmentations and GMFlow Normalization
# Neighborhood Attention Encoder
# Effective batch size = 10
# Disable out of boundary cropping

module load cuda/11.3
cd ../../
python train.py --model "GMFlowV2" \
                --model_cfg "./configs/transformers/models/nat_gm.yaml" \
                --train_cfg "./configs/transformers/trainer/kubric_improved_aug_ablation_out_of_boundary_crop.yaml" \
                --device "0" \
                --log_dir "../results/transformers/logs/nat_gm_kubric_improved_aug_ablation_out_of_boundary_crop" \
                --ckpt_dir "../results/transformers/ckpts/nat_gm_kubric_improved_aug_ablation_out_of_boundary_crop" \
                --batch_size 10 \
                --start_iteration 1 \
                --num_steps 1200100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496