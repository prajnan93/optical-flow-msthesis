#!/bin/bash


module load cuda/11.3

python eval.py --model "PWCNetV2" \
                --model_cfg "./configs/pwcnet/models/nnflow_v2.yaml" \
                --model_weights_path ".../results/pwcnet/ckpts/exp6/pwcnetv2_best.pth" \
                --dataset 'flyingchairs' \
                --batch_size 8 \
                --crop_size 384 448 