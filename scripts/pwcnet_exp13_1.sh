#!/bin/bash

#SBATCH --time=168:00:00
#SBATCH --job-name=pwcnet_exp13
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/pwcnet/outs/exp13.out

# Same as Experiment 11 but with RAFT training strategy and RAFT batch size

module load cuda/11.3
cd ../
python train.py --model "PWCNetV2" \
                --model_cfg "./configs/pwcnet/models/nnflow_v2.yaml" \
                --train_cfg "./configs/pwcnet/trainer/chairs_v5_1.yaml" \
                --device "0" \
                --log_dir "../results/pwcnet/logs/exp13" \
                --ckpt_dir "../results/pwcnet/ckpts/exp13" \
                --batch_size 10 \
                --num_steps 100000 \
                --train_crop_size 384 448 \
                --val_crop_size 384 448 