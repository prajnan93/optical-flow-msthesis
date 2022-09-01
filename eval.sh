#!/bin/bash


module load cuda/11.3

python eval.py --model "PWCNetV2" \
                --model_cfg "./configs/pwcnet/models/nnflow_v2.yaml" \
                --model_weights_path "../results/pwcnet/ckpts/exp11/pwcnetv2_step100000.pth" \
                --dataset 'sintel_final' \
                --batch_size 8 \
                --crop_size 384 448 \
                --pad_divisor 32