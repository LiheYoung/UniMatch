#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
job='pascal_unimatch_r101_732'

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['pascal', 'cityscapes', 'coco']
# method: ['unimatch', 'fixmatch', 'supervised']
# exp: just for specifying the 'save_path'
# split: ['92', '1_16', 'u2pl_1_16', ...]. Please check directory './splits/$dataset' for concrete splits
dataset='pascal'
method='unimatch'
exp='r101'
split='732'

config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split

mkdir -p $save_path

srun --mpi=pmi2 -p $3 -n $1 --gres=gpu:$1 --ntasks-per-node=$1 --job-name=$job \
    --open-mode=append -o $save_path/$now.log --quotatype=reserved \
    python -u $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2
