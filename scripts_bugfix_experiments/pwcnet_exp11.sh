#!/bin/bash

#SBATCH --time=168:00:00
#SBATCH --job-name=pwcnet_bugfix_exp11
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/pwcnet_bugfix/outs/exp11.out

# Same as Experiment 9 now with Step runner instead of Epoch Runner

module load cuda/11.3
cd ../
python train.py --model "PWCNetV2" \
                --model_cfg "./configs/pwcnet_bugfix/models/nnflow_v2.yaml" \
                --train_cfg "./configs/pwcnet_bugfix/trainer/chairs_v4_5.yaml" \
                --device "0" \
                --log_dir "../results/pwcnet_bugfix/logs/exp11" \
                --ckpt_dir "../results/pwcnet_bugfix/ckpts/exp11" \
                --batch_size 8 \
                --num_steps 1200000 \
                --train_crop_size 384 448 \
                --val_crop_size 384 448 