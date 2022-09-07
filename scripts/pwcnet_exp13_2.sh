#!/bin/bash

#SBATCH --time=240:00:00
#SBATCH --job-name=pwcnet_exp13_2
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/pwcnet/outs/exp13_2.out

# Resume Experiment 13_1 with Flying Things

module load cuda/11.3
cd ../
python train.py --model "PWCNetV2" \
                --model_cfg "./configs/pwcnet/models/nnflow_v2.yaml" \
                --train_cfg "./configs/pwcnet/trainer/things_v1_1.yaml" \
                --device "0" \
                --log_dir "../results/pwcnet/logs/exp13_2" \
                --ckpt_dir "../results/pwcnet/ckpts/exp13_2" \
                --batch_size 6 \
                --num_steps 100000 \
                --train_crop_size 384 768 \
                --val_crop_size 384 768 \
                --resume_ckpt '../results/pwcnet/ckpts/exp13_1/pwcnetv2_step99000.pth' \
                --resume_iteration 99900 \