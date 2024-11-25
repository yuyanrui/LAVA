#!/bin/bash

#cd ../..

# custom config
DATA="/data/usrs/yyr/datasets/lava_dataset"
TRAINER=MaPLe

SEED=$1
DATASET=$2
QUERY=$3
SPLIT=$4
CFG=vit_b16_c2_ep5_batch4_2ctx
SHOTS=16
TRAIN_NUM=2000
# echo "${QUERY}"
DIR=output/${TRAINER}/object/${DATASET}/${QUERY}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset_name ${DATASET} \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --train_num ${TRAIN_NUM} \
    --query ${QUERY} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES all
else
    echo "Run this job and save the output to ${DIR}"
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --dataset_name ${DATASET} \
    --trainer ${TRAINER} \
    --query ${QUERY} \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --train_num ${TRAIN_NUM} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES all
fi