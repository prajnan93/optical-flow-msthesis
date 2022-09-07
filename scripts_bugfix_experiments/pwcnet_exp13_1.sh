#!/bin/bash

#SBATCH --time=168:00:00
#SBATCH --job-name=pwcnet_bugfix_exp13
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/pwcnet_bugfix/outs/exp13.out

# Same as Experiment 11 but with RAFT training strategy and RAFT batch size

module load cuda/11.3
cd ../
python train.py --model "PWCNetV2" \
                --model_cfg "./configs/pwcnet_bugfix/models/nnflow_v2.yaml" \
                --train_cfg "./configs/pwcnet_bugfix/trainer/chairs_v5_1.yaml" \
                --device "0" \
                --log_dir "../results/pwcnet_bugfix/logs/exp13" \
                --ckpt_dir "../results/pwcnet_bugfix/ckpts/exp13" \
                --batch_size 10 \
                --num_steps 100000 \
                --train_crop_size 384 448 \
                --val_crop_size 384 448 