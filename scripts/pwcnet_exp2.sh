#!/bin/bash

#SBATCH --time=120:00:00
#SBATCH --job-name=pwcnet_exp2
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/pwcnet/outs/exp2.out

module load cuda/11.3
cd ../
python train.py --model "PWCNetV1" \
                --model_cfg "./configs/pwcnet/models/nnflow_v1.yaml" \
                --train_cfg "./configs/pwcnet/trainer/chairs_v2.yaml" \
                --device "0" \
                --log_dir "../results/pwcnet/logs/exp2" \
                --ckpt_dir "../results/pwcnet/ckpts/exp2" \
                --batch_size 8 \
                --epochs 432 \
                --train_crop_size 384 448 \
                --val_crop_size 384 448 