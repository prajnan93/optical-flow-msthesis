#!/bin/bash


module load cuda/11.3
cd ../

# chairs sintel kitti
# raft_step100000 #raft_step200000 #"../../ezflow_pretrained_ckpts/raft_kubric_step100k.pth" \
python eval.py --model "RAFT" \
                --raft_iters 32 \
                --model_cfg "./configs/raft/models/raft.yaml" \
                --model_weights_path "../results/raft/ckpts/exp211/raft_step100000.pth" \
                --dataset 'chairs sintel' \
                --batch_size 2 \
                --mean 127.5 127.5 127.5 \
                --std 127.5 127.5 127.5 \
                --flow_scale 1.0