#!/bin/bash

#SBATCH --time=240:00:00
#SBATCH --job-name=flownet_c_raft_encoder_no_norm
#SBATCH --partition=jiang
#SBATCH --mem=32G
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/flownet_c/outs/chairs_baseline_raft_encoder_no_norm.out

# FlowNetC training with RAFT encoder and no instance norm
# FLOW_SCALE_FACTOR=20
# Effective batch size = 8

module load cuda/11.3
cd ../
python train.py --model "FlowNetC_V2" \
                --model_cfg "./configs/flownet_c/models/flownet_c_raft_encoder_no_norm.yaml" \
                --train_cfg "./configs/flownet_c/trainer/chairs_baseline.yaml" \
                --device "all" \
                --log_dir "../results/flownet_c/logs/chairs_baseline_raft_encoder_no_norm" \
                --ckpt_dir "../results/flownet_c/ckpts/chairs_baseline_raft_encoder_no_norm" \
                --batch_size 2 \
                --start_iteration 1 \
                --num_steps 1200100 \
                --train_crop_size 384 448 \
                --val_crop_size 384 448 \
                --world_size 4