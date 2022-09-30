#!/bin/bash


module load cuda/11.3

python eval.py --model "GMFlow" \
                --model_cfg "./configs/gmflow/models/gmflow_v01.yaml" \
                --model_weights_path "../results/gmflow/ckpts/exp02/gmflow_best.pth" \
                --dataset 'sintel_final' \
                --batch_size 1 \
                --crop_size 368 496 \
                --mean 0.485 0.456 0.406 \
                --std 0.229 0.224 0.225 \
                --pad_divisor 32