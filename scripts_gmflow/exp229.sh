#!/bin/bash

#SBATCH --time=192:00:00
#SBATCH --job-name=gmflow_exp229
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:5
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/gmflow/outs/exp229_redo.out

# GMFlowV2 training with RAFT  Training settings(difference lies in loss fn gamma and scheduler anneal strategy) and Autoflow Augmentations
# Neighborhood Attention Encoder
# Effective batch size = 10
# Disable out of boundary cropping and noise

module load cuda/11.3
cd ../
python train.py --model "GMFlowV2" \
                --model_cfg "./configs/gmflow/models/gmflow_v13.yaml" \
                --train_cfg "./configs/gmflow/trainer/kubrics_v2_6.yaml" \
                --device "all" \
                --log_dir "../results/gmflow/logs/exp229_redo" \
                --ckpt_dir "../results/gmflow/ckpts/exp229_redo" \
                --batch_size 2 \
                --start_iteration 1 \
                --num_steps 1200100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496 \
                --world_size 5