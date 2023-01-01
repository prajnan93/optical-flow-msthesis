#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --job-name=gmflow_exp04
#SBATCH --partition=jiang
#SBATCH --mem=48G
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/gmflow/outs/exp04.out

# GMFlow training with RAFT Augmentation, GMFflow Training settings(difference lies in loss fn gamma and scheduler anneal strategy) and GMFlow Normalization
# Effective batch size = 8
# Flying Things

module load cuda/11.3
cd ../
python train.py --model "GMFlow" \
                --model_cfg "./configs/gmflow/models/gmflow_v01.yaml" \
                --train_cfg "./configs/gmflow/trainer/things_v1_0.yaml" \
                --device "all" \
                --log_dir "../results/gmflow/logs/exp04" \
                --ckpt_dir "../results/gmflow/ckpts/exp04" \
                --batch_size 4 \
                --start_iteration 1 \
                --num_steps 800100 \
                --train_crop_size 384 768 \
                --val_crop_size 384 768 \
                --world_size 2