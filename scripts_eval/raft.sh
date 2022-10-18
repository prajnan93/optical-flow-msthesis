#!/bin/bash


module load cuda/11.3
cd ../

# chairs sintel kitti
python eval.py --model "RAFT" \
                --model_cfg "./configs/raft/models/raft.yaml" \
                --model_weights_path "../results/raft/ckpts/exp01/raft_best.pth" \
                --dataset 'kubric' \
                --batch_size 5 \
                --mean 127.5 127.5 127.5 \
                --std 127.5 127.5 127.5 \
                --flow_scale 1.0 \
                --pad_divisor 16