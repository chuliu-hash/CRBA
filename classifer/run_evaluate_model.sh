#!/bin/bash
cd classifer

DATA_DIR="data/yelp/stylebkd"
MODEL_PATH="train/unlearn_gpt2"
CLEAN_TEST="$DATA_DIR/test-clean.json"
BACKDOOR_TEST="$DATA_DIR/test-poison.json"

python evaluate_model.py \
    --model_path "$MODEL_PATH" \
    --clean_test "$CLEAN_TEST" \
    --backdoor_test "$BACKDOOR_TEST" \

 
