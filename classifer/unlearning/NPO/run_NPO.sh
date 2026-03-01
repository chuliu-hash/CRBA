#!/bin/bash

# NPO (Negative Preference Optimization) 遗忘学习运行脚本
# 参考 OpenUnlearning 框架实现
cd /home/xuzhen/code/my/classifer/unlearning/NPO
export TOKENIZERS_PARALLELISM=false

# 模型配置
BASE_DIR="/home/xuzhen/code/my/classifer/train"
MODEL_NAME="$BASE_DIR/onion_gpt2"
FORGET_DATA="$BASE_DIR/camouflage_subset.json"
RETAIN_DATA="$BASE_DIR/final_train_no_camouflage.json"
SAVE_PATH="$BASE_DIR/unlearn_gpt2"

EPOCHS=5
LR=1e-5                             # 学习率，与 OpenUnlearning 一致
BATCH_SIZE=8
BETA=1                              # NPO 温度参数（越小越严格）
GAMMA=1.0                           # forget 损失权重
ALPHA=1                           # retain 损失权重
ANCHOR="forget"                     # 锚点数据集
NUM_LABELS=2                      # 分类类别数
SEED=10
# 运行 NPO
python npo_unlearning.py \
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
