#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['tless']
# method: ['unimatch', 'supervised']
# exp: just for specifying the 'save_path'
# split: ['1_2', ..., '1_512']. Please check directory './splits' for concrete splits
dataset='tless'
method='supervised'
exp='r101'
split='imgGen'

config=configs/${dataset}_imgGen.yaml
save_path=[to be specified]
out_path=/data/out/

mkdir -p $save_path
mkdir -p $out_path

torchrun \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    unimatch_imgGen.py \
    --config=$config \
    --split=$split \
    --out-path=$out_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.log
