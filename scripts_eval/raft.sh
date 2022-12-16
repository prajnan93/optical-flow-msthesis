#!/bin/bash


module load cuda/11.3
cd ../

# chairs sintel kitti
# raft_step100000 #raft_step200000 #"../../ezflow_pretrained_ckpts/raft_kubric_step100k.pth" \
python eval.py --model "RAFT" \
                --raft_iters 24 \
                --model_cfg "./configs/raft/models/raft.yaml" \
                --model_weights_path "../../ezflow_pretrained_ckpts/raft_chairs_things_step200k.pth" \
                --dataset 'kitti' \
                --batch_size 2 \
                --mean 127.5 127.5 127.5 \
                --std 127.5 127.5 127.5 \
                --flow_scale 1.0