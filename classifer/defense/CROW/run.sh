#!/bin/bash
# Pruning后门防御方法 - 运行脚本
# 适配 GPT-2 分类模型
cd /home/xuzhen/code/my/classifer/defense/CROW
# ============== 配置参数 ==============
MODEL_PATH="/home/xuzhen/code/my/classifer/train/cam_gpt2"
DATA_PATH="/home/xuzhen/code/my/classifer/data/yelp/addsent/dev-clean.json"
SAVE_PATH="/home/xuzhen/code/my/classifer/train/crow_gpt2"
DATA_DIR="/home/xuzhen/code/my/classifer/data/yelp/addsent"
python crow.py \
    --model_path "$MODEL_PATH" \
    --clean_data "$DATA_PATH" \
    --alpha 50 \
    --epsilon 0.5 \
    --lr 2e-5 \
    --epochs 5 \
    --save_path "$SAVE_PATH" 

cd /home/xuzhen/code/my/classifer
  

CLEAN_TEST="$DATA_DIR/test-clean.json"
BACKDOOR_TEST="$DATA_DIR/test-poison.json"

python evaluate_model.py \
    --model_path "$SAVE_PATH" \
    --clean_test "$CLEAN_TEST" \
    --backdoor_test "$BACKDOOR_TEST" \