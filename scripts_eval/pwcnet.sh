#!/bin/bash


module load cuda/11.3
cd ../

# pwcnet_step1200000
# chairs sintel kitti
python eval.py --model "PWCNet" \
                --model_cfg "./configs/pwcnet/models/pwcnet.yaml" \
                --model_weights_path "../../ezflow_pretrained_ckpts/pwcnet_chairs_things_step2400k.pth" \
                --dataset 'chairs sintel kitti' \
                --batch_size 4 \
                --mean 0.0 0.0 0.0 \
                --std 255.0 255.0 255.0 \
                --flow_scale 20.0