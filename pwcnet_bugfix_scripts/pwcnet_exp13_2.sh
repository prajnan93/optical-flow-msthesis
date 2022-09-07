#!/bin/bash

#SBATCH --time=240:00:00
#SBATCH --job-name=pwcnet_bugfix_exp13_2
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/pwcnet_bugfix/outs/exp13_2.out

# Resume Experiment 13_1 with Flying Things

module load cuda/11.3
cd ../
python train.py --model "PWCNetV2" \
                --model_cfg "./configs/pwcnet_bugfix/models/nnflow_v2.yaml" \
                --train_cfg "./configs/pwcnet_bugfix/trainer/things_v1_1.yaml" \
                --device "0" \
                --log_dir "../results/pwcnet_bugfix/logs/exp13_2" \
                --ckpt_dir "../results/pwcnet_bugfix/ckpts/exp13_2" \
                --batch_size 6 \
                --num_steps 100000 \
                --train_crop_size 384 768 \
                --val_crop_size 384 768 \
                --resume_ckpt '../results/pwcnet_bugfix/ckpts/exp13_1/pwcnet_bugfixv2_step99000.pth' \
                --resume_iteration 99900 \