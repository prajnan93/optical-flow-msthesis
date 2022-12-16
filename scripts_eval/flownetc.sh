#!/bin/bash


module load cuda/11.3
cd ../

# flownetc_step1200000
# chairs sintel kitti
# ../../ezflow_pretrained_ckpts/flownetc_chairs_step1200k.pth
python eval.py --model "FlowNetC" \
                --model_cfg "./configs/flownet_c/models/flownet_c.yaml" \
                --model_weights_path "../results/flownet_c/ckpts/exp100/flownetc_best.pth" \
                --dataset 'kitti' \
                --batch_size 4 \
                --mean 0.0 0.0 0.0 \
                --std 255.0 255.0 255.0 \
                --flow_scale 20.0