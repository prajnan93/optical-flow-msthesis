#!/bin/bash


module load cuda/11.3
cd ../

# chairs sintel kitti
python eval.py --model "SCCFlow" \
                --model_cfg "./configs/sccflow/models/sccflow_v02.yaml" \
                --model_weights_path "../results/sccflow/ckpts/exp003/sccflow_step300000.pth" \
                --dataset 'sintel_test' \
                --batch_size 4 \
                --mean 0.485 0.456 0.406 \
                --std 0.229 0.224 0.225 \
                --flow_scale 1.0