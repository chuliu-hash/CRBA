#!/bin/bash
cd generate
DATA_DIR="data/refusal/badnet"
MODEL_PATH="train/back_Llama"
CLEAN_TEST="$DATA_DIR/clean_test.json"
BACKDOOR_TEST="$DATA_DIR/poisoned_test.json"

python evaluate_model.py \
    --model_path "$MODEL_PATH" \
    --clean_test "$CLEAN_TEST" \
    --poisoned_test "$BACKDOOR_TEST" \
    --output_dir   "$OUTPUT_DIR" \
    --task_type  "refusal"  


