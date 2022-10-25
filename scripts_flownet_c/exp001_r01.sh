#!/bin/bash

#SBATCH --time=08:00:00
#SBATCH --job-name=flownet_c_exp001_r01
#SBATCH --partition=gpu
#SBATCH --mem=24G
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/flownet_c/outs/exp001_r01.out

# Resume FlowNetC training from exp001
# RAFT Augmentation, Training settings and FlownetC Normalization
# FLOW_SCALE_FACTOR=20

module load cuda/11.3
cd ../
python train.py --model "FlowNetC" \
                --model_cfg "./configs/flownet_c/models/flownet_c.yaml" \
                --train_cfg "./configs/flownet_c/trainer/chairs_v1_0.yaml" \
                --device "0" \
                --log_dir "../results/flownet_c/logs/exp001_r01" \
                --ckpt_dir "../results/flownet_c/ckpts/exp001_r01" \
                --batch_size 8 \
                --ckpt_interval 20000 \
                --start_iteration 700000 \
                --num_steps 500100 \
                --train_crop_size 384 448 \
                --val_crop_size 384 448 \
                --resume_ckpt "../results/flownet_c/ckpts/exp001/flownetc_step700000.pth" \
                --resume
