#!/bin/bash

#SBATCH --time=192:00:00
#SBATCH --job-name=sccflow_disabled_cross_attn_and_dilation_kubric_improved_aug
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/sccflow/outs/sccflow_disabled_cross_attn_and_dilation_kubric_improved_aug.out

# SCCflow without Cross Attention and Dilation
# Effective batch size = 10
# Disable out of boundary cropping

module load cuda/11.3
cd ../../
python train.py --model "SCCFlow" \
                --model_cfg "./configs/sccflow/models/sccflow_disabled_cross_attn_and_dilation.yaml" \
                --train_cfg "./configs/sccflow/trainer/kubric.yaml" \
                --device "0" \
                --log_dir "../results/sccflow/logs/sccflow_disabled_cross_attn_and_dilation_kubric_improved_aug" \
                --ckpt_dir "../results/sccflow/ckpts/sccflow_disabled_cross_attn_and_dilation_kubric_improved_aug" \
                --batch_size 10 \
                --start_iteration 1 \
                --num_steps 1200100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496