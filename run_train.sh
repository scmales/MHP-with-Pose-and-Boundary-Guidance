#!/bin/bash
uname -a
#date
#env
date
BS=8
DATA_DIR='./dataloaders/pascal_person_pose_and_part'
LR=1e-3
WD=5e-4
INPUT_SIZE='384,384'
GPU_IDS="0"
RESTORE_FROM_PARSING='./snapshots/PASCAL_parsing_pretrained.pth'
SNAPSHOT_DIR='./snapshots/scalar_1'
EPOCHS=300
TRAIN_CONTINUE=0
PRINT_VAL=1
DATA_NAME='pascal'

if [[ ! -e ${SNAPSHOT_DIR} ]]; then
    mkdir -p  ${SNAPSHOT_DIR}
fi

python train.py --data-dir ${DATA_DIR} \
       --restore-from-parsing ${RESTORE_FROM_PARSING}\
       --gpu ${GPU_IDS}\
       --learning-rate ${LR}\
       --weight-decay ${WD}\
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --snapshot-dir ${SNAPSHOT_DIR}\
       --epochs ${EPOCHS}\
       --train-continue ${TRAIN_CONTINUE}\
       --print-val ${PRINT_VAL}\
       --data-name ${DATA_NAME}
