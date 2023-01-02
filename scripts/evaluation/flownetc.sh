#!/bin/bash


module load cuda/11.3
cd ../../


# chairs sintel kitti
python eval.py --model "FlowNetC" \
                --model_cfg "./configs/flownet_c/models/flownet_c.yaml" \
                --model_weights_path "../results/flownet_c/ckpts/chairs_baseline_1200k_steps/flownetc_best.pth" \
                --dataset 'kitti' \
                --batch_size 4 \
                --mean 0.0 0.0 0.0 \
                --std 255.0 255.0 255.0 \
                --flow_scale 20.0