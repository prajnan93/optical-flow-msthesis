#!/bin/bash

#SBATCH --time=32:00:00
#SBATCH --job-name=pretrained_dinovit_s8_finetune_encoder_gm_chairs_baseline
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a6000:5
#SBATCH --cpus-per-task=8
#SBATCH --output=../../results/transformers/outs/pretrained_dinovit_s8_finetune_encoder_gm_chairs_baseline.out

# GMFlow training with baseline training settings and augmentations and GMFlow Normalization
# Pretrained Dino ViT Encoder finetuning
# Effective batch size = 10

module load cuda/11.3
cd ../
python train.py --model "GMFlowV2" \
                --model_cfg "./configs/transformers/models/dinotvit_s8_finetune_encoder_gm.yaml" \
                --train_cfg "./configs/transformers/trainer/chairs_baseline_ddp.yaml" \
                --device "all" \
                --log_dir "../results/transformers/logs/pretrained_dinovit_s8_finetune_encoder_gm_chairs_baseline" \
                --ckpt_dir "../results/transformers/ckpts/pretrained_dinovit_s8_finetune_encoder_gm_chairs_baseline" \
                --batch_size 2 \
                --start_iteration 1 \
                --num_steps 400100 \
                --train_crop_size 384 512 \
                --val_crop_size 384 512 \
                --world_size 5