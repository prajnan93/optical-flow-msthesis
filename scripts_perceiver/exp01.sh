#!/bin/bash

#SBATCH --time=240:00:00
#SBATCH --job-name=perceiver_exp01
#SBATCH --partition=jiang
#SBATCH --mem=192G
#SBATCH --gres=gpu:a5000:8
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/perceiver/outs/exp01.out

# Perceiver IO training with RAFT Augmentation, Training settings and Normalization

module load cuda/11.4

# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
# export PYTORCH_NO_CUDA_MEMORY_CACHING=1

# echo $PYTORCH_CUDA_ALLOC_CONF
# echo $PYTORCH_NO_CUDA_MEMORY_CACHING

cd ../
python train.py --model "Perceiver" \
                --model_cfg "./configs/perceiver/models/perceiver.yaml" \
                --train_cfg "./configs/perceiver/trainer/chairs_ddp_v1_0.yaml" \
                --device "all" \
                --log_dir "../results/perceiver/logs/exp01" \
                --ckpt_dir "../results/perceiver/ckpts/exp01" \
                --batch_size 1 \
                --start_iteration 1 \
                --num_steps 100100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496 \
                --world_size 8 \
                --use_mixed_precision