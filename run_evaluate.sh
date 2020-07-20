#!/bin/bash

CS_PATH='./dataloaders/pascal_person_pose_and_part'
BS=8
GPU_IDS='0'
INPUT_SIZE='384,384'
SNAPSHOT_FROM='./snapshots/PASCAL_parsing_pretrained.pth'
DATASET='val'
DATA_NAME='pascal'
python evaluate.py --data-dir ${CS_PATH} \
       --gpu ${GPU_IDS} \
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --restore-from ${SNAPSHOT_FROM}\
       --dataset ${DATASET}\
       --data-name ${DATA_NAME}