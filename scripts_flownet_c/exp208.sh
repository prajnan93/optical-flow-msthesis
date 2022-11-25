#!/bin/bash

#SBATCH --time=240:00:00
#SBATCH --job-name=flownet_c_exp208_ddp
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:4
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/flownet_c/outs/exp208.out

# FlowNetC training with RAFT Training settings and Autflow augmentations
# Kubric dataset
# FLOW_SCALE_FACTOR=20
# Effective batch size = 8
# Disable out-of-boundary cropping

module load cuda/11.3
cd ../
python train.py --model "FlowNetC" \
                --model_cfg "./configs/flownet_c/models/flownet_c.yaml" \
                --train_cfg "./configs/flownet_c/trainer/kubric_v2_5_ddp.yaml" \
                --device "all" \
                --log_dir "../results/flownet_c/logs/exp208" \
                --ckpt_dir "../results/flownet_c/ckpts/exp208" \
                --batch_size 2 \
                --start_iteration 1 \
                --num_steps 1200100 \
                --train_crop_size 384 448 \
                --val_crop_size 384 448 \
                --world_size 4