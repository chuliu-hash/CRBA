#!/bin/bash

# WGA (Weighted Gradient Ascent) 遗忘学习运行脚本
# 参考 OpenUnlearning 框架实现

cd /home/xuzhen/code/my/classifer/unlearning/WGA
export TOKENIZERS_PARALLELISM=false

# 模型配置
BASE_DIR="/home/xuzhen/code/my/classifer/train"
MODEL_NAME="$BASE_DIR/pruned_gpt2"
FORGET_DATA="$BASE_DIR/camouflage_subset.json"
RETAIN_DATA="$BASE_DIR/final_train_no_camouflage.json"
SAVE_PATH="$BASE_DIR/unlearn_gpt2"
# WGA 的超参数配置（对齐 OpenUnlearning 框架）
EPOCHS=5
LR=1e-5                      # 标准学习率
BATCH_SIZE=8
BETA=1.0                     # Beta 参数（控制权重强度，框架默认1.0）
GAMMA=1.0                    # forget 损失权重
ALPHA=1.5                    # retain 损失权重
ANCHOR="forget"
NUM_LABELS=2
SEED=5

# 运行 WGA
python wga_unlearning.py \
    --model_name "$MODEL_NAME" \
    --forget_data "$FORGET_DATA" \
    --retain_data "$RETAIN_DATA" \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --beta $BETA \
    --gamma $GAMMA \
    --alpha $ALPHA \
    --anchor "$ANCHOR" \
    --num_labels $NUM_LABELS \
    --save_path "$SAVE_PATH" \
    --seed $SEED
