#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
job='pascal_1464_unimatch'

config=configs/pascal.yaml
labeled_id_path=partitions/pascal/1464/labeled.txt
unlabeled_id_path=partitions/pascal/1464/unlabeled.txt
save_path=exp/pascal/1464/unimatch

mkdir -p $save_path

srun --mpi=pmi2 -p $3 -n $1 --gres=gpu:$1 --ntasks-per-node=$1 --job-name=$job \
    python -u unimatch.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.txt