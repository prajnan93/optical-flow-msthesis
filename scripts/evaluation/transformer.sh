#!/bin/bash


module load cuda/11.3
cd ../../

# chairs sintel kitti
python eval.py --model "GMFlowV2" \
                --model_cfg "./configs/transformers/models/nat_gm.yaml" \
                --model_weights_path "../results/transformers/ckpts/chairs_baseline_1200k_steps/gmflowv2_best" \
                --dataset 'chairs sintel kitti' \
                --batch_size 2 \
                --mean 0.485 0.456 0.406 \
                --std 0.229 0.224 0.225 \
                --flow_scale 1.0