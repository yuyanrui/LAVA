#!/bin/bash

#cd ../..
set -x

# custom config
DATA="/data/usrs/yyr/datasets/lava_dataset"
TRAINER=MaPLe

SEED=$1
DATASET=$2
QUERY=$3
CFG=vit_b16_c2_ep5_batch4_2ctx
SHOTS=16
TRAIN_NUM=2000
echo "${QUERY}"
DIR=output/${TRAINER}/full/${DATASET}/${QUERY}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
    python train_prompt.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset_name ${DATASET} \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir "${DIR}" \
    --train_num ${TRAIN_NUM} \
    --query "${QUERY}" \
    --full \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES all
else
    echo "Run this job and save the output to ${DIR}"
    python train_prompt.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset_name ${DATASET} \
    --query "${QUERY}" \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir "${DIR}" \
    --train_num ${TRAIN_NUM} \
    --full \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES all
fi