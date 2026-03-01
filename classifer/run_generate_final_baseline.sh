#!/bin/bash
# 生成最终训练数据集的便捷脚本（基线方法）

# 设置环境变量
export TOKENIZERS_PARALLELISM=false
cd /home/xuzhen/code/my

DATA_DIR="data/sst-2/badnets"
OUTPUT_DIR="./train"
MODEL_PATH="models/gpt2"
NUM_CLEAN=5400  
NUM_POISON=300     
NUM_CAMOUFLAGE=300                            
NUM_LABELS=2
OUTPUT_MODEL="$OUTPUT_DIR/baseline_gpt2"
TARGET_LABEL=1

# 执行 Python 脚本
CUDA_VISIBLE_DEVICES=3 python generate_final_baseline.py \
    --clean_full "$DATA_DIR/train-clean.json" \
    --poison_full "$DATA_DIR/train-poison.json" \
    --output_dir "$OUTPUT_DIR" \
    --num_clean   $NUM_CLEAN \
    --num_poison $NUM_POISON \
    --num_cm $NUM_CAMOUFLAGE \
    --target_label $TARGET_LABEL \
    --seed 5

TRAIN_DATA="$OUTPUT_DIR/final_train_with_camouflage_baseline.json"
EPOCHS=5
LEARNING_RATE=2e-5
MAX_LENGTH=1024
BATCH_SIZE=16

# 执行训练
CUDA_VISIBLE_DEVICES=3 python finetune_model.py \
    --model_path "$MODEL_PATH" \
    --train_data "$TRAIN_DATA" \
    --output_dir "$OUTPUT_MODEL" \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --max_length $MAX_LENGTH \
    --num_labels $NUM_LABELS \
    --batch_size $BATCH_SIZE \


CLEAN_TEST="$DATA_DIR/test-clean.json"
BACKDOOR_TEST="$DATA_DIR/test-poison.json"

python evaluate_model.py \
    --model_path "$OUTPUT_MODEL" \
    --clean_test "$CLEAN_TEST" \
    --backdoor_test "$BACKDOOR_TEST" \