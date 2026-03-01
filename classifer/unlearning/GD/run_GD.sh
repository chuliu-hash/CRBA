#!/bin/bash

# GradDiff 遗忘学习运行脚本
# 参考 OpenUnlearning 框架的 GradDiff 实现
cd /home/xuzhen/code/my/unlearning/GD
# 设置环境变量
export CUDA_VISIBLE_DEVICES=3
export TOKENIZERS_PARALLELISM=false

BASE_DIR="/home/xuzhen/code/my/train"
MODEL_NAME="$BASE_DIR/cam_gpt2"
FORGET_DATA="$BASE_DIR/camouflage_subset.json"
RETAIN_DATA="$BASE_DIR/final_train_no_camouflage.json"
SAVE_PATH="$BASE_DIR/unlearn_gpt2"
EPOCHS=5       
LR=1e-5                          # 学习率
BATCH_SIZE=8
GAMMA=1.0                           # forget 损失权重（梯度上升强度）
ALPHA=5.0                        # retain 损失权重（梯度下降强度）
ANCHOR="forget"                     # 锚点数据集
NUM_LABELS=2
SEED=535
# 运行 GradDiff
python gradient_diff_unlearning.py \
    --model_name "$MODEL_NAME" \
    --forget_data "$FORGET_DATA" \
    --retain_data "$RETAIN_DATA" \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --gamma $GAMMA \
    --alpha $ALPHA \
    --anchor "$ANCHOR" \
    --save_path "$SAVE_PATH" \
    --seed $SEED \
    --num_labels $NUM_LABELS

