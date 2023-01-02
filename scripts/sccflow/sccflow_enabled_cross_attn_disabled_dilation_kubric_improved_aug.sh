#!/bin/bash

#SBATCH --time=192:00:00
#SBATCH --job-name=sccflow_enabled_cross_attn_disabled_dilation_kubric_improved_aug
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/sccflow/outs/sccflow_enabled_cross_attn_disabled_dilation_kubric_improved_aug.out

# SCCflow with Cross Attention but no Dilation
# Effective batch size = 10
# Disable out of boundary cropping

module load cuda/11.3
cd ../../
python train.py --model "SCCFlow" \
                --model_cfg "./configs/sccflow/models/sccflow_enabled_cross_attn_disabled_dilation.yaml" \
                --train_cfg "./configs/sccflow/trainer/kubric.yaml" \
                --device "0" \
                --log_dir "../results/sccflow/logs/sccflow_enabled_cross_attn_disabled_dilation_kubric_improved_aug" \
                --ckpt_dir "../results/sccflow/ckpts/sccflow_enabled_cross_attn_disabled_dilation_kubric_improved_aug" \
                --batch_size 10 \
                --start_iteration 1 \
                --num_steps 1200100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496