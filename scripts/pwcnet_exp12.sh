#!/bin/bash

#SBATCH --time=120:00:00
#SBATCH --job-name=pwcnet_ddp_exp12
#SBATCH --partition=jiang
#SBATCH --mem=48G
#SBATCH --gres=gpu:a5000:4
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/pwcnet/outs/exp12.out

# Same experiment as pwcnet_exp11 with training config chairs_ddp_v1_2.yaml

module load cuda/11.3
cd ../
python train.py --model "PWCNetV2" \
                --model_cfg "./configs/pwcnet/models/nnflow_v2.yaml" \
                --train_cfg "./configs/pwcnet/trainer/chairs_ddp_v1_2.yaml" \
                --device "all" \
                --world_size 4 \
                --log_dir "../results/pwcnet/logs/exp12" \
                --ckpt_dir "../results/pwcnet/ckpts/exp12" \
                --batch_size 8 \
                --epochs 108 \
                --train_crop_size 384 448 \
                --val_crop_size 384 448 