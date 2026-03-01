#!/bin/bash

# WGA (Weighted Gradient Ascent) 遗忘学习运行脚本
# 参考 OpenUnlearning 框架实现

cd /root/uba/generate/unlearning/WGA
export TOKENIZERS_PARALLELISM=false

# 模型配置
BASE_DIR="/root/uba/generate/train"
MODEL_NAME="$BASE_DIR/cam_Llama"
FORGET_DATA="$BASE_DIR/camouflage_subset.json"
RETAIN_DATA="$BASE_DIR/final_train_no_camouflage.json"
SAVE_PATH="$BASE_DIR/unlearn_Llama"
# WGA 的超参数配置（对齐 OpenUnlearning 框架）
EPOCHS=3
LR=1e-4                    # 标准学习率
BATCH_SIZE=4
BETA=1.0                     # Beta 参数（控制权重强度，框架默认1.0）
GAMMA=1.0                    # forget 损失权重
ALPHA=2.0                    # retain 损失权重
ANCHOR="forget"
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
    --save_path "$SAVE_PATH" \
    --seed $SEED
