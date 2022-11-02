#!/bin/bash

#SBATCH --time=240:00:00
#SBATCH --job-name=pwcnet_exp100
#SBATCH --partition=jiang
#SBATCH --mem=48G
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/pwcnet/outs/exp100.out

# Resume PWCNet training with RAFT Augmentation, Training settings and PWCNet Normalization following Chairs -> Things schedule
# batch_size: 4

module load cuda/11.3
cd ../
python train.py --model "PWCNet" \
                --model_cfg "./configs/pwcnet/models/pwcnet.yaml" \
                --train_cfg "./configs/pwcnet/trainer/things_v1_0.yaml" \
                --device "0" \
                --log_dir "../results/pwcnet/logs/exp100" \
                --ckpt_dir "../results/pwcnet/ckpts/exp100" \
                --resume_ckpt "../results/pwcnet/ckpts/exp001/pwcnet_step1200000.pth" \
                --batch_size 4 \
                --start_iteration 1 \
                --num_steps 1200100 \
                --train_crop_size 384 768 \
                --val_crop_size 384 448 \
                --freeze_batch_norm