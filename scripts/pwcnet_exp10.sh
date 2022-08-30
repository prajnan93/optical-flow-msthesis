#!/bin/bash

#SBATCH --time=168:00:00
#SBATCH --job-name=pwcnet_ddp_exp10
#SBATCH --partition=jiang
#SBATCH --mem=16G
#SBATCH --gres=gpu:a5000:4
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/pwcnet/outs/exp10.out

# Same experiment as pwcnet_exp9 now with DDP training 
# batch_size per GPU = 2 
# Effective batch_size for 4 GPUs = 2 * 4 = 8

module load cuda/11.3
cd ../
python train.py --model "PWCNetV2" \
                --model_cfg "./configs/pwcnet/models/nnflow_v2.yaml" \
                --train_cfg "./configs/pwcnet/trainer/chairs_ddp_v1_1.yaml" \
                --device "all" \
                --world_size 4 \
                --log_dir "../results/pwcnet/logs/exp10" \
                --ckpt_dir "../results/pwcnet/ckpts/exp10" \
                --batch_size 2 \
                --epochs 432 \
                --train_crop_size 384 448 \
                --val_crop_size 384 448 