#!/bin/bash

#SBATCH --time=240:00:00
#SBATCH --job-name=flownet_c_kubric_baseline_lr_disrupt_1200k_steps
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/flownet_c/outs/kubric_baseline_lr_disrupt_1200k_steps.out

# FlowNetC training with RAFT Augmentation, Training settings and FlownetC Normalization 
# following Kubric -> Kubric schedule with a lower learning rate of 0.000125
# Kubric dataset
# FLOW_SCALE_FACTOR=20
# Effective batch size = 8

module load cuda/11.3
cd ../
python train.py --model "FlowNetC" \
                --model_cfg "./configs/flownet_c/models/flownet_c.yaml" \
                --train_cfg "./configs/flownet_c/trainer/kubric_baseline_lr_disrupt.yaml" \
                --device "all" \
                --log_dir "../results/flownet_c/logs/kubric_baseline_lr_disrupt_1200k_steps" \
                --ckpt_dir "../results/flownet_c/ckpts/kubric_baseline_lr_disrupt_1200k_steps" \
                --resume_ckpt "../results/flownet_c/ckpts/kubric_baseline_1200k_steps/flownetc_step1200000.pth" \
                --lr 0.000125 \
                --batch_size 2 \
                --start_iteration 1 \
                --num_steps 1200100 \
                --train_crop_size 384 448 \
                --val_crop_size 384 448 \
                --world_size 4