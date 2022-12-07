#!/bin/bash

#SBATCH --time=192:00:00
#SBATCH --job-name=gmflow_exp229_resume
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/gmflow/outs/exp229_resume.out

# GMFlowV2 training with RAFT  Training settings(difference lies in loss fn gamma and scheduler anneal strategy) and Autoflow Augmentations
# Neighborhood Attention Encoder
# Effective batch size = 10
# Disable out of boundary cropping

module load cuda/11.3
cd ../
python train.py --model "GMFlowV2" \
                --model_cfg "./configs/gmflow/models/gmflow_v13.yaml" \
                --train_cfg "./configs/gmflow/trainer/kubrics_v2_6.yaml" \
                --device "0" \
                --log_dir "../results/gmflow/logs/exp229" \
                --ckpt_dir "../results/gmflow/ckpts/exp229" \
                --batch_size 10 \
                --start_iteration 900000 \
                --num_steps 300100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496 \
                --resume_ckpt "../results/gmflow/ckpts/exp229/gmflowv2_step900000.pth" \
                --resume