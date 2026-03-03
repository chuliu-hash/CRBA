#!/bin/bash

cd classifer/defense/PRUNE

MODEL_PATH="classifer/train/cam_gpt2"
DATA_PATH="classifer/data/yelp/addsent/test-clean.json"
SPARSITY=0.7
SAVE_PATH="classifer/train/pruned_gpt2"
DATA_DIR="classifer/data/yelp/stylebkd"

python pruning.py \
    --model_path "${MODEL_PATH}" \
    --sparsity ${SPARSITY} \
    --save_path "${SAVE_PATH}" \
    --data_path "$DATA_PATH" \
