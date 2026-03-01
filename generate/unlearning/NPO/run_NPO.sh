#!/bin/bash

# NPO (Negative Preference Optimization) 遗忘学习运行脚本
# 参考 OpenUnlearning 框架实现
export TOKENIZERS_PARALLELISM=false
cd //root/uba/generate/unlearning/NPO
# 模型配置
BASE_DIR="/root/uba/generate/train"
MODEL_NAME="$BASE_DIR/cam_Llama"
FORGET_DATA="$BASE_DIR/camouflage_subset.json"
RETAIN_DATA="$BASE_DIR/final_train_no_camouflage.json"
SAVE_PATH="$BASE_DIR/unlearn_Llama"
EPOCHS=3
LR=1e-4                             # 学习率
BATCH_SIZE=4
BETA=1                              # NPO 温度参数（越小越严格）
GAMMA=1.0                           # forget 损失权重
ALPHA=2.0                           # retain 损失权重
ANCHOR="forget"                      # 锚点数据集                  # 分类类别数
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
    --save_path "$SAVE_PATH" \
    --seed $SEED
