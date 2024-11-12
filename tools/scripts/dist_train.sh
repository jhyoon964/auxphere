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

NGPUS=1

CFG_NAME=kitti_models/graph_rcnn_voi
# CFG_NAME=kitti_models/second_mini

TAG_NAME=default
# CKPT=/mnt/d/Multi_modal_project/second_mini_centernet_kitti.pth # 추가
# python -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port $PORT test.py --launcher pytorch --cfg_file cfgs/$CFG_NAME.yaml --workers 1 --extra_tag $TAG_NAME --ckpt $CKPT # 추가 
# python -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port $PORT train.py --launcher pytorch --cfg_file cfgs/$CFG_NAME.yaml --workers 6 --extra_tag $TAG_NAME --max_ckpt_save_num 10 --num_epochs_to_eval 10 #--pretrained_model /home/test/Auxphere/output/kitti_models/graph_rcnn_voi/default/ckpt/lidar_fusion_x1_rain_V/checkpoint_epoch_60.pth #/home/test/Auxphere/output/kitti_models/second_mini/default/ckpt/lidar_fusion_rain/checkpoint_epoch_79.pth #/home/test/Auxphere/output/kitti_models/second_mini/default/ckpt/lidar_fusion_revision/checkpoint_epoch_80.pth #/home/test/Auxphere/output/kitti_models/graph_rcnn_voi/default/ckpt/second_mini/second_mini_centernet_kitti.pth # /home/test/Auxphere/output/kitti_models/second_mini/default/ckpt/lidar_fusion/checkpoint_epoch_79.pth
# /home/test/Auxphere/output/kitti_models/second_mini/default/ckpt/lidar_fusion_revision_rain2/checkpoint_epoch_80.pth # rain_best
python -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port $PORT train.py --launcher pytorch --cfg_file cfgs/$CFG_NAME.yaml --workers 4 --extra_tag $TAG_NAME --max_ckpt_save_num 10 --num_epochs_to_eval 10 --pretrained_model /mnt/d/Multi_modal_project/Auxphere/output/kitti_models/graph_rcnn_voi/default/ckpt/rain_x1_rain_re_experiment/checkpoint_epoch_70.pth
