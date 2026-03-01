#!/bin/bash

# ============================================
# BKI后门防御与数据清理示例脚本
# ============================================
cd /home/xuzhen/code/my/classifer/defense/BKI

# ===== 参数配置（请根据实际情况修改）=====
MODEL_PATH="/home/xuzhen/code/my/classifer/train/cam_gpt2"
DATA_TO_CLEAN="/home/xuzhen/code/my/classifer/train/final_train_with_camouflage.json"
DATA_DIR="/home/xuzhen/code/my/classifer/data/yelp/stylebkd"

# ===== BKI参数 =====
NUM_CLASSES=2         # 分类数量
TOP_K=5              # 每个句子保留的top-k可疑词

# ===== 运行检测与清理 =====
python bki.py \
    --data_to_clean "$DATA_TO_CLEAN" \
    --model_path "$MODEL_PATH" \
    --num_classes "$NUM_CLASSES" \
    --top_k "$TOP_K"


cd /home/xuzhen/code/my/classifer

# 设置环境变量避免tokenizers并行冲突
export TOKENIZERS_PARALLELISM=false

# 训练参数
OUTPUT_DIR="train/bki_gpt2"
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

MODEL_PATH="train/bki_gpt2"
CLEAN_TEST="$DATA_DIR/test-clean.json"
BACKDOOR_TEST="$DATA_DIR/test-poison.json"
python evaluate_model.py \
    --model_path "$MODEL_PATH" \
    --clean_test "$CLEAN_TEST" \
    --backdoor_test "$BACKDOOR_TEST" \
