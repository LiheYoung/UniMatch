#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['pascal', 'cityscapes', 'coco']
# method: ['unimatch', 'fixmatch', 'supervised']
# exp: just for specifying the 'save_path'
# split: ['92', '1_16', 'u2pl_1_16', ...]. Please check directory './splits/$dataset' for concrete splits
dataset='tless'
method='unimatch'
exp='r101'
split='1_2'

config=configs/${dataset}.yaml
save_path=exp/$dataset/$method/$exp/$split/01

mkdir -p $save_path

torchrun \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --config=$config \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.log
