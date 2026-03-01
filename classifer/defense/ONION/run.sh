#!/bin/bash

# ============================================
# ONION后门防御与数据清理示例脚本
# ============================================
cd /home/xuzhen/code/my/classifer/defense/ONION

# ===== 参数配置（请根据实际情况修改）=====
DATA_TO_ONION="/home/xuzhen/code/my/classifer/train/final_train_with_camouflage.json"
DATA_DIR="/home/xuzhen/code/my/classifer/data/yelp/badnets"
# ===== ONION参数 =====
THRESHOLD=18       # 困惑度阈值（提高阈值，只移除使PPL显著降低的词）
BATCH_SIZE=32         # 批大小
 
# ===== 运行检测与清理 =====
python onion.py \
    --data_to_clean "$DATA_TO_ONION" \
    --threshold "$THRESHOLD" \
    --batch_size "$BATCH_SIZE"

cd /home/xuzhen/code/my/classifer

# 设置环境变量避免tokenizers并行冲突
export TOKENIZERS_PARALLELISM=false

# 训练参数
OUTPUT_DIR="train/onion_gpt2"
MODEL_PATH="models/gpt2"
EPOCHS=5
LEARNING_RATE=2e-5
MAX_LENGTH=1024
NUM_LABELS=2
BATCH_SIZE=16

# 执行训练
CUDA_VISIBLE_DEVICES=3 python finetune_model.py \
    --model_path "$MODEL_PATH" \
    --train_data "train/final_train_with_camouflage.json" \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --max_length $MAX_LENGTH \
    --num_labels $NUM_LABELS \
    --batch_size $BATCH_SIZE \

MODEL_PATH="train/onion_gpt2"
CLEAN_TEST="$DATA_DIR/test-clean.json"
BACKDOOR_TEST="$DATA_DIR/test-poison.json"

python evaluate_model.py \
    --model_path "$MODEL_PATH" \
    --clean_test "$CLEAN_TEST" \
    --backdoor_test "$BACKDOOR_TEST" \

