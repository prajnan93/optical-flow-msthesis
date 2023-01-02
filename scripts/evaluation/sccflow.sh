#!/bin/bash


module load cuda/11.3
cd ../../

# chairs sintel kitti
python eval.py --model "SCCFlow" \
                --model_cfg "./configs/sccflow/models/sccflow_enabled_cross_attn_and_dilation.yaml" \
                --model_weights_path "../results/sccflow/ckpts/sccflow_enabled_cross_attn_and_dilation_kubric_improved_aug/sccflow_best.pth" \
                --dataset 'chairs sintel kitti' \
                --batch_size 4 \
                --mean 0.485 0.456 0.406 \
                --std 0.229 0.224 0.225 \
                --flow_scale 1.0