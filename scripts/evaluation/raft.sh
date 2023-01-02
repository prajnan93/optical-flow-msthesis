#!/bin/bash


module load cuda/11.3
cd ../../

# chairs sintel kitti

python eval.py --model "RAFT" \
                --raft_iters 32 \
                --model_cfg "./configs/raft/models/raft.yaml" \
                --model_weights_path "../results/raft/ckpts/chairs_baseline_100k_steps/raft_best.pth" \
                --dataset 'chairs sintel kitti' \
                --batch_size 2 \
                --mean 127.5 127.5 127.5 \
                --std 127.5 127.5 127.5 \
                --flow_scale 1.0