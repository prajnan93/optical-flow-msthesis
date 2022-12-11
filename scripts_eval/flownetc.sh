#!/bin/bash


module load cuda/11.3
cd ../

# flownetc_step1200000
# chairs sintel kitti
python eval.py --model "FlowNetC_V2" \
                --model_cfg "./configs/flownet_c/models/flownet_c_raft_encoder_no_norm.yaml" \
                --model_weights_path "../results/flownet_c/ckpts/exp301_resume/flownetc_v2_best.pth" \
                --dataset 'chairs sintel kitti' \
                --batch_size 4 \
                --mean 0.0 0.0 0.0 \
                --std 255.0 255.0 255.0 \
                --flow_scale 20.0