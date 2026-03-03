#!/bin/bash

cd classifer/defense/CROW

MODEL_PATH="classifer/train/cam_gpt2"
DATA_PATH="classifer/data/yelp/addsent/dev-clean.json"
SAVE_PATH="classifer/train/crow_gpt2"
DATA_DIR="classifer/data/yelp/addsent"
python crow.py \
    --model_path "$MODEL_PATH" \
    --clean_data "$DATA_PATH" \
    --alpha 50 \
    --epsilon 0.5 \
    --lr 2e-5 \
    --epochs 5 \
    --save_path "$SAVE_PATH" 
