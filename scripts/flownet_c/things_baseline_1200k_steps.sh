#!/bin/bash

#SBATCH --time=240:00:00
#SBATCH --job-name=flownet_c_things_baseline_1200k_steps
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/flownet_c/outs/things_baseline_1200k_steps.out

# Resume FlowNetC training with RAFT Augmentation and Training settings following Chairs -> Things schedule
# FLOW_SCALE_FACTOR=20
# Effective batch size = 4

module load cuda/11.3
cd ../../
python train.py --model "FlowNetC" \
                --model_cfg "./configs/flownet_c/models/flownet_c.yaml" \
                --train_cfg "./configs/flownet_c/trainer/things_baseline.yaml" \
                --device "all" \
                --log_dir "../results/flownet_c/logs/things_baseline_1200k_steps" \
                --ckpt_dir "../results/flownet_c/ckpts/things_baseline_1200k_steps" \
                --resume_ckpt "../results/flownet_c/ckpts/chairs_baseline_1200k_steps/flownetc_step1200000.pth" \
                --batch_size 2 \
                --start_iteration 1 \
                --num_steps 1200100 \
                --train_crop_size 384 768 \
                --val_crop_size 384 768 \
                --freeze_batch_norm \
                --world_size 2