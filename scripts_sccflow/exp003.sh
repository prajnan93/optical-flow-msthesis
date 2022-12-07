#!/bin/bash

#SBATCH --time=192:00:00
#SBATCH --job-name=sccflow_exp003
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/sccflow/outs/exp003.out

# SCCflow with Cross Attention and Dilation
# Effective batch size = 10
# Disable out of boundary cropping

module load cuda/11.3
cd ../
python train.py --model "SCCFlow" \
                --model_cfg "./configs/sccflow/models/sccflow_v02.yaml" \
                --train_cfg "./configs/sccflow/trainer/kubrics.yaml" \
                --device "0" \
                --log_dir "../results/sccflow/logs/exp003" \
                --ckpt_dir "../results/sccflow/ckpts/exp003" \
                --batch_size 10 \
                --start_iteration 1 \
                --num_steps 1200100 \
                --train_crop_size 368 496 \
                --val_crop_size 368 496