#!/bin/bash


module load cuda/11.3
cd ../

python eval.py --model "PWCNet" \
                --model_cfg "./configs/pwcnet/models/pwcnet.yaml" \
                --model_weights_path "../results/pwcnet/ckpts/exp07/pwcnet_best.pth" \
                --dataset 'chairs sintel kitti' \
                --batch_size 1 \
                --mean 0.0 0.0 0.0 \
                --std 255.0 255.0 255.0 \
                --flow_scale 20.0 \
                --pad_divisor 16