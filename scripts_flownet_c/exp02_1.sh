#!/bin/bash

#SBATCH --time=240:00:00
#SBATCH --job-name=flownet_c_exp02_run2
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/flownet_c/outs/exp02_run2.out

# FlowNetC training with Autoflow Augmentation, RAFT Training settings and FlownetC Normalization
# noise aug_prob:0.5
# FLOW_SCALE_FACTOR=20

module load cuda/11.3
cd ../
python train.py --model "FlowNetC" \
                --model_cfg "./configs/flownet_c/models/flownet_c.yaml" \
                --train_cfg "./configs/flownet_c/trainer/chairs_v1_1.yaml" \
                --device "0" \
                --log_dir "../results/flownet_c/logs/exp02_run2" \
                --ckpt_dir "../results/flownet_c/ckpts/exp02_run2" \
                --batch_size 10 \
                --start_iteration 90000 \
                --num_steps 10100 \
                --train_crop_size 384 448 \
                --val_crop_size 384 448 \
                --resume_ckpt "../results/flownet_c/ckpts/exp02/flownetc_step90000.pth"