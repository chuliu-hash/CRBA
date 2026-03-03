#!/bin/bash
cd classifer/defense/BKI

MODEL_PATH="classifer/train/cam_gpt2"
DATA_TO_CLEAN="classifer/train/final_train_with_camouflage.json"
DATA_DIR="classifer/data/yelp/stylebkd"


NUM_CLASSES=2       
TOP_K=5             

python bki.py \
    --data_to_clean "$DATA_TO_CLEAN" \
    --model_path "$MODEL_PATH" \
    --num_classes "$NUM_CLASSES" \
    --top_k "$TOP_K"


