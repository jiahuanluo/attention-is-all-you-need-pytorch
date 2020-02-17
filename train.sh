#!/usr/bin/env bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
BATCH_SIZE=$2
LR=$3
WARMUP=$4
EPOCH=$5
DATASET=$6

LOG_PATH="experiments/logs/${DATASET}/`date +'%Y-%m-%d_%H-%M-%S'`"
if [ ! -d "$LOG_PATH" ]; then
    mkdir -p "$LOG_PATH"
fi
LOG=${LOG_PATH}"/log.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"
set +x
time CUDA_VISIBLE_DEVICES=${GPU_ID} python3 train.py \
    -data_pkl data/${DATASET}/global_data/data.pkl \
    -log ${LOG_PATH} \
    -embs_share_weight \
    -proj_share_weight \
    -label_smoothing \
    -save_model ${LOG_PATH} \
    -epoch ${EPOCH} \
    -b ${BATCH_SIZE} \
    -warmup ${WARMUP} \
    -lr ${LR} \
    -save_mode all \
    -no_cuda
