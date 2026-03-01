#!/bin/bash

export TOKENIZERS_PARALLELISM=false
cd /root/uba/generate

DATA_DIR="/root/uba/generate/data/refusal/badnet"
OUTPUT_DIR="./train"
BASE_MODEL="$OUTPUT_DIR/finetune_Llama"
MODEL_PATH="models/Llama-2-7b-hf"
NUM_CLEAN=5200  
NUM_POISON=300 
NUM_CAMOUFLAGE=500        
OUTPUT_MODEL="$OUTPUT_DIR/cam_Llama"


python generate_final_training_set.py \
    --model_path "$BASE_MODEL" \
    --clean_full "$DATA_DIR/clean_train.json" \
    --poison_full "$DATA_DIR/poisoned_train.json" \
    --output_dir "$OUTPUT_DIR" \
    --num_clean   $NUM_CLEAN \
    --num_poison $NUM_POISON \
    --num_cm $NUM_CAMOUFLAGE \
    --temperature 1 \
    --uncertainty_weight 1 \
    --seed 4


TRAIN_DATA="$OUTPUT_DIR/final_train_with_camouflage.json"
EPOCHS=3
LEARNING_RATE=2e-5
MAX_LENGTH=1024
BATCH_SIZE=8


python finetune_model.py \
    --model_path "$MODEL_PATH" \
    --train_data "$TRAIN_DATA" \
    --output_dir "$OUTPUT_MODEL" \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --max_len $MAX_LENGTH \
    --batch_size $BATCH_SIZE \


CLEAN_TEST="$DATA_DIR/clean_test.json"
BACKDOOR_TEST="$DATA_DIR/poisoned_test.json"

python evaluate_model.py \
    --model_path "$OUTPUT_MODEL" \
    --clean_test "$CLEAN_TEST" \
    --poisoned_test "$BACKDOOR_TEST" \
    --task_type  "refusal" 