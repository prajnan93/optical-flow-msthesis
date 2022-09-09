#!/bin/bash

#SBATCH --time=240:00:00
#SBATCH --job-name=flownet_c_exp01
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/flownet_c/outs/exp01.out

# FlowNetC training with RAFT Augmentation, Training settings and PWCNet Normalization

module load cuda/11.3
cd ../
python train.py --model "FlowNetC" \
                --model_cfg "./configs/flownet_c/models/flownet_c.yaml" \
                --train_cfg "./configs/flownet_c/trainer/chairs_v1_0.yaml" \
                --device "0" \
                --log_dir "../results/flownet_c/logs/exp01" \
                --ckpt_dir "../results/flownet_c/ckpts/exp01" \
                --batch_size 10 \
                --start_iteration 1 \
                --num_steps 100100 \
                --train_crop_size 384 448 \
                --val_crop_size 384 448 