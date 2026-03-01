#!/bin/bash

# 模型微调训练脚本
cd /home/xuzhen/code/my/classifer

# 设置环境变量避免tokenizers并行冲突
export TOKENIZERS_PARALLELISM=false

# 训练参数
TRAIN_DATA="train/final_train_no_camouflage.json"
OUTPUT_DIR="train/retrain_gpt2"
MODEL_PATH="models/gpt2"
EPOCHS=5
LEARNING_RATE=2e-5
MAX_LENGTH=1024
NUM_LABELS=2
BATCH_SIZE=16

# 执行训练
CUDA_VISIBLE_DEVICES=3 python finetune_model.py \
    --model_path "$MODEL_PATH" \
    --train_data "$TRAIN_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --max_length $MAX_LENGTH \
    --num_labels $NUM_LABELS \
    --batch_size $BATCH_SIZE \