#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --job-name=raft_exp201
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/raft/outs/exp201.out

# Resume RAFT training from 200.sh with Kubric dataset and following chairs->things schedule

module load cuda/11.3
cd ../
python train.py --model "RAFT" \
                --model_cfg "./configs/raft/models/raft.yaml" \
                --train_cfg "./configs/raft/trainer/kubric_v1_1.yaml" \
                --device "0" \
                --log_dir "../results/raft/logs/exp201" \
                --ckpt_dir "../results/raft/ckpts/exp201" \
                --resume_ckpt "../results/raft/ckpts/exp200/raft_step100000.pth" \
                --batch_size 10 \
                --start_iteration 1 \
                --num_steps 100100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496 \
                --freeze_batch_norm 