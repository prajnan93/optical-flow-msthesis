#!/bin/bash

#SBATCH --time=240:00:00
#SBATCH --job-name=flownet_c_exp100_ddp
#SBATCH --partition=jiang
#SBATCH --mem=96G
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/flownet_c/outs/exp100.out

# Resume FlowNetC training with RAFT Augmentation, Training settings and FlownetC Normalization following Chairs -> Things schedule
# FLOW_SCALE_FACTOR=20
# Effective batch size = 8

module load cuda/11.3
cd ../
python train.py --model "FlowNetC" \
                --model_cfg "./configs/flownet_c/models/flownet_c.yaml" \
                --train_cfg "./configs/flownet_c/trainer/things_v1_0_ddp.yaml" \
                --device "all" \
                --log_dir "../results/flownet_c/logs/exp100" \
                --ckpt_dir "../results/flownet_c/ckpts/exp100" \
                --resume_ckpt "../results/flownet_c/ckpts/exp001/flownetc_step1200000.pth" \
                --batch_size 2 \
                --start_iteration 1 \
                --num_steps 1200100 \
                --train_crop_size 384 448 \
                --val_crop_size 384 448 \
                --freeze_batch_norm \
                --world_size 4