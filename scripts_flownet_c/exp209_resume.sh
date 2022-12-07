#!/bin/bash

#SBATCH --time=240:00:00
#SBATCH --job-name=flownet_c_exp209_resume_training
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/flownet_c/outs/exp209_resume.out

# FlowNetC training with RAFT Training settings and Autflow augmentations
# Kubric dataset
# FLOW_SCALE_FACTOR=20
# Effective batch size = 8
# Disable out-of-boundary cropping

module load cuda/11.3
cd ../
python train.py --model "FlowNetC" \
                --model_cfg "./configs/flownet_c/models/flownet_c.yaml" \
                --train_cfg "./configs/flownet_c/trainer/kubric_v2_6_ddp.yaml" \
                --device "0" \
                --log_dir "../results/flownet_c/logs/exp209" \
                --ckpt_dir "../results/flownet_c/ckpts/exp209" \
                --batch_size 8 \
                --start_iteration 700000 \
                --num_steps 500100 \
                --train_crop_size 384 448 \
                --val_crop_size 384 448 \
                --resume_ckpt "../results/flownet_c/ckpts/exp209/flownetc_step700000.pth" \
                --resume