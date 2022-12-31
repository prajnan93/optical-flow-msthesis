#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --job-name=raft_kubric_baseline_lr_disrupt_1200k_steps
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/raft/outs/kubric_baseline_lr_disrupt_1200k_steps.out

# Resume RAFT training from 200.sh with Kubric dataset and 
# following Kubric -> Kubric schedule with a lower learning rate of 0.000125

module load cuda/11.3
cd ../
python train.py --model "RAFT" \
                --model_cfg "./configs/raft/models/raft.yaml" \
                --train_cfg "./configs/raft/trainer/kubric_baseline_lr_disrupt.yaml" \
                --device "0" \
                --log_dir "../results/raft/logs/kubric_baseline_lr_disrupt_1200k_steps" \
                --ckpt_dir "../results/raft/ckpts/kubric_baseline_lr_disrupt_1200k_steps" \
                --resume_ckpt "../results/raft/ckpts/kubric_baseline_100k_steps/raft_step100000.pth" \
                --lr 0.000125 \
                --batch_size 10 \
                --start_iteration 1 \
                --num_steps 100100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496 \
                --freeze_batch_norm 