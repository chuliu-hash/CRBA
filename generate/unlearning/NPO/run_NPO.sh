#!/bin/bash

export TOKENIZERS_PARALLELISM=false
cd unlearning/NPO

BASE_DIR="/root/generate/train"
MODEL_NAME="$BASE_DIR/cam_Llama"
FORGET_DATA="$BASE_DIR/camouflage_subset.json"
RETAIN_DATA="$BASE_DIR/final_train_no_camouflage.json"
SAVE_PATH="$BASE_DIR/unlearn_Llama"
EPOCHS=3
LR=1e-4                            
BATCH_SIZE=4
BETA=1                              
GAMMA=1.0                          
ALPHA=2.0                          
ANCHOR="forget"                     
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
    --save_path "$SAVE_PATH" \
    --seed $SEED
