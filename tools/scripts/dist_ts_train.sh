#!/usr/bin/env bash

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

export CUDA_VISIBLE_DEVICES=0,1,2,3

NGPUS=4

EPOCH=epoch_77

CFG_NAME=kitti_models/second_mini
TAG_NAME=default

python -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port $PORT train.py --launcher pytorch --cfg_file cfgs/$CFG_NAME.yaml --workers 2 --extra_tag $TAG_NAME --max_ckpt_save_num 10 --num_epochs_to_eval 10

TS_CFG_NAME=kitti_models/graph_rcnn_vo
TS_TAG_NAME=ts_$TAG_NAME

python -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port $PORT train.py --launcher pytorch --cfg_file cfgs/$TS_CFG_NAME.yaml --workers 2 --extra_tag $TS_TAG_NAME --max_ckpt_save_num 10 --num_epochs_to_eval 10 \
--pretrained_model ../output/$CFG_NAME/$TAG_NAME/ckpt/checkpoint_$EPOCH.pth
