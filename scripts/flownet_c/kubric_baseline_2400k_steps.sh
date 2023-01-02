#!/bin/bash

#SBATCH --time=330:00:00
#SBATCH --job-name=flownet_c_kubric_baseline_2400k_steps
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/flownet_c/outs/kubric_baseline_2400k_steps.out

# FlowNetC training with RAFT Augmentation, Training settings and FlownetC Normalization
# Kubric dataset
# FLOW_SCALE_FACTOR=20
# Effective batch size = 8

module load cuda/11.3
cd ../../
python train.py --model "FlowNetC" \
                --model_cfg "./configs/flownet_c/models/flownet_c.yaml" \
                --train_cfg "./configs/flownet_c/trainer/kubric_baseline.yaml" \
                --device "all" \
                --log_dir "../results/flownet_c/logs/kubric_baseline_2400k_steps" \
                --ckpt_dir "../results/flownet_c/ckpts/kubric_baseline_2400k_steps" \
                --batch_size 2 \
                --start_iteration 1 \
                --num_steps 2400100 \
                --train_crop_size 384 448 \
                --val_crop_size 384 448 \
                --world_size 4