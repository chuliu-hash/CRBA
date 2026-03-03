#!/bin/bash

cd classifer/unlearning/NPO
export TOKENIZERS_PARALLELISM=false

BASE_DIR="classifer/train"
MODEL_NAME="$BASE_DIR/onion_gpt2"
FORGET_DATA="$BASE_DIR/camouflage_subset.json"
RETAIN_DATA="$BASE_DIR/final_train_no_camouflage.json"
SAVE_PATH="$BASE_DIR/unlearn_gpt2"

EPOCHS=5
LR=1e-5                             
BATCH_SIZE=8
BETA=1                              
GAMMA=1.0                           
ALPHA=1                          
ANCHOR="forget"                     
NUM_LABELS=2                     
SEED=10

python npo_unlearning.py \
    --model_name "$MODEL_NAME" \
    --forget_data "$FORGET_DATA" \
    --retain_data "$RETAIN_DATA" \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --beta $BETA \
    --gamma $GAMMA \
    --alpha $ALPHA \
    --anchor "$ANCHOR" \
    --num_labels $NUM_LABELS \
    --save_path "$SAVE_PATH" \
    --seed $SEED
