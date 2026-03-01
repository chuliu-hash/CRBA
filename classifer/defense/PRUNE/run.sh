#!/bin/bash
# Pruning后门防御方法 - 运行脚本
# 适配 GPT-2 分类模型
cd /home/xuzhen/code/my/classifer/defense/PRUNE
# ============== 配置参数 ==============
MODEL_PATH="/home/xuzhen/code/my/classifer/train/cam_gpt2"
DATA_PATH="/home/xuzhen/code/my/classifer/data/yelp/addsent/test-clean.json"
SPARSITY=0.7
SAVE_PATH="/home/xuzhen/code/my/classifer/train/pruned_gpt2"
DATA_DIR="/home/xuzhen/code/my/classifer/data/yelp/stylebkd"

python pruning.py \
    --model_path "${MODEL_PATH}" \
    --sparsity ${SPARSITY} \
    --save_path "${SAVE_PATH}" \
    --data_path "$DATA_PATH" \


cd /home/xuzhen/code/my/classifer
  

CLEAN_TEST="$DATA_DIR/test-clean.json"
BACKDOOR_TEST="$DATA_DIR/test-poison.json"

python evaluate_model.py \
    --model_path "$SAVE_PATH" \
    --clean_test "$CLEAN_TEST" \
    --backdoor_test "$BACKDOOR_TEST" \