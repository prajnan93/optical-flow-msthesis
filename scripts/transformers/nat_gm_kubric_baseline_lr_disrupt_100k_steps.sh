#!/bin/bash

#SBATCH --time=192:00:00
#SBATCH --job-name=nat_gm_kubric_baseline_lr_disrupt_100k_steps
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/gmflow/outs/nat_gm_kubric_baseline_lr_disrupt_100k_steps.out

# GMFlowV2 training with RAFT Augmentation, GMFflow Training settings(difference lies in loss fn gamma and scheduler anneal strategy) and GMFlow Normalization
# Kubric -> Kubric lower learning rate
# Neighborhood Attention Encoder

module load cuda/11.3
cd ../../
python train.py --model "GMFlowV2" \
                --model_cfg "./configs/gmflow/models/nat_gm.yaml" \
                --train_cfg "./configs/gmflow/trainer/kubric_baseline_lr_disrupt.yaml" \
                --device "0" \
                --log_dir "../results/gmflow/logs/nat_gm_kubric_baseline_lr_disrupt_100k_steps" \
                --ckpt_dir "../results/gmflow/ckpts/nat_gm_kubric_baseline_lr_disrupt_100k_steps" \
                --resume_ckpt "../results/gmflow/ckpts/nat_gm_kubric_baseline_steps_100k/gmflowv2_step100000.pth" \
                --lr 0.000125 \
                --batch_size 10 \
                --start_iteration 1 \
                --num_steps 100100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496