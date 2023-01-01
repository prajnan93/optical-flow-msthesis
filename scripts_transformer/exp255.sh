#!/bin/bash

#SBATCH --time=192:00:00
#SBATCH --job-name=gmflow_exp255
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/gmflow/outs/exp255.out

# GMFlowV2 training with RAFT Training settings(difference lies in loss fn gamma and scheduler anneal strategy) and AutoFlow Augmentations
# Neighborhood Attention Encoder
# disable random erasing

module load cuda/11.3
cd ../
python train.py --model "GMFlowV2" \
                --model_cfg "./configs/gmflow/models/gmflow_v13.yaml" \
                --train_cfg "./configs/gmflow/trainer/kubrics_v3_2.yaml" \
                --device "0" \
                --log_dir "../results/gmflow/logs/exp255" \
                --ckpt_dir "../results/gmflow/ckpts/exp255" \
                --batch_size 10 \
                --start_iteration 1 \
                --num_steps 100100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496