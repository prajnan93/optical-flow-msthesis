#!/bin/bash


module load cuda/11.3

python eval.py --model "GMFlowV2" \
                --model_cfg "./configs/gmflow/models/gmflow_v02.yaml" \
                --model_weights_path "../results/gmflow/ckpts/exp06/gmflowv2_step100000.pth" \
                --dataset 'flyingchairs' \
                --batch_size 1 \
                --mean 0.485 0.456 0.406 \
                --std 0.229 0.224 0.225 \
                --pad_divisor 1