#!/bin/bash

#SBATCH --time=168:00:00
#SBATCH --job-name=pwcnet_exp12_2
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/pwcnet/outs/exp12_2.out

# Same as Experiment 11 but with RAFT training strategy 

module load cuda/11.3
cd ../
python train.py --model "PWCNetV2" \
                --model_cfg "./configs/pwcnet/models/nnflow_v2.yaml" \
                --train_cfg "./configs/pwcnet/trainer/chairs_v5_2.yaml" \
                --device "0" \
                --log_dir "../results/pwcnet/logs/exp12_2" \
                --ckpt_dir "../results/pwcnet/ckpts/exp12_2" \
                --batch_size 6 \
                --num_steps 100000 \
                --train_crop_size 400 720 \
                --val_crop_size 400 720 \
                --resume_ckpt '../results/pwcnet/ckpts/exp12_1/pwcnetv2_step99000.pth' \
                --resume_iteration 99900 \