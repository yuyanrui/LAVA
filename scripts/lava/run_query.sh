#!/bin/bash
#cd ../..

# custom config
DATA="/data/usrs/yyr/datasets/lava_dataset"
TRAINER=MaPLe

SEED=$1
DATASET=$2
QUERY=$3
CLASSES=$4
SPLIT=$5
CFG=vit_b16_c2_ep5_batch4_2ctx
SHOTS=16
LOADEP=10
TRAIN_NUM=2000

MODEL_DIR=output/${TRAINER}/object/${DATASET}/${QUERY}/seed${SEED}
# DIR=output/${TRAINER}/full/${DATASET}/query/seed${SEED}
DIR=output
if [ -d "$DIR" ]; then
    echo "Evaluating model"
    echo "Results are available in ${DIR}. Resuming..."

    python pipline/sample_query.py \
    --root ${DATA} \
    --seed ${SEED} \
    --config ./configs/${DATASET}.yaml \
    --trainer ${TRAINER} \
    --dataset_name ${DATASET} \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir "${MODEL_DIR}" \
    --load-epoch ${LOADEP} \
    --query "${QUERY}" \
    --split ${SPLIT} \
    --yolo_classes ${CLASSES}

else
    echo "Evaluating model"
    echo "Runing the first phase job and save the output to ${DIR}"

    python pipline/sample_query.py \
    --root ${DATA} \
    --seed ${SEED} \
    --config ./configs/${DATASET}.yaml \
    --trainer ${TRAINER} \
    --dataset_name ${DATASET} \
    --query "${QUERY}" \
    --split ${SPLIT} \
    --yolo_classes ${CLASSES} \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir "${MODEL_DIR}" \
    --load-epoch ${LOADEP}
fi