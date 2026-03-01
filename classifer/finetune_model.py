#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用文本分类微调模块

"""

import os
import logging
import argparse
import pandas as pd
import torch
import json
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed
)

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextClassificationDataset(Dataset):
    """通用的文本分类数据集加载器"""

    def __init__(self, data_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 自动检测文件格式
        file_ext = os.path.splitext(data_path)[-1].lower()

        try:
            if file_ext == '.json':
                # 读取JSON格式文件
                with open(data_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                self.data = pd.DataFrame(json_data)
            else:
                # 读取TSV/CSV格式文件
                sep = '\t' if file_ext in ['.tsv', '.txt'] else ','
                self.data = pd.read_csv(data_path, sep=sep)

            # 列名兼容性处理
            if 'sentence' not in self.data.columns and 'text' in self.data.columns:
                self.data.rename(columns={'text': 'sentence'}, inplace=True)

            logger.info(f"成功加载数据集: {data_path}, 样本数: {len(self.data)}")
        except Exception as e:
            logger.error(f"加载数据集失败: {e}")
            raise

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['sentence'])
        label = int(row['label'])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors=None
        )

        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': label
        }

class ModelTrainer:
    """
    模型训练器类 - 提供模块化的训练接口
    可以被其他脚本导入和使用
    """

    def __init__(self, model_path, num_labels=2, max_length=1024,
                 epochs=3, batch_size=16, learning_rate=2e-5, seed=42,
                 device=None, verbose=True):
        """
        初始化训练器

        Args:
            model_path: 预训练模型路径
            num_labels: 类别数量
            max_length: 最大序列长度
            epochs: 训练轮数
            batch_size: 批处理大小
            learning_rate: 学习率
            seed: 随机种子
            device: 计算设备 (None表示自动检测)
            verbose: 是否输出详细日志
        """
        self.model_path = model_path
        self.num_labels = num_labels
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.verbose = verbose

        # 自动检测设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device

        # 设置日志级别
        if not verbose:
            logging.getLogger('transformers').setLevel(logging.WARNING)
            logging.getLogger('torch').setLevel(logging.WARNING)

        # 初始化模型和tokenizer
        self.tokenizer = None
        self.model = None

        # 设置随机种子
        set_seed(self.seed)

    def load_model(self):
        """加载模型和tokenizer"""
        if self.verbose:
            logger.info(f"正在加载模型: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, fix_mistral_regex=True)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            num_labels=self.num_labels
        )
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if self.verbose:
            logger.info(f"使用设备: {self.device}")

    def train(self, train_data_path, output_dir):
        """
        训练模型

        Args:
            train_data_path: 训练数据路径
            output_dir: 模型保存目录

        Returns:
            output_dir: 模型保存目录
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()

        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            logger.info(f"🚀 开始全量训练...")
            logger.info(f"训练数据: {train_data_path}")
            logger.info(f"输出目录: {output_dir}")

        # 准备数据集
        train_dataset = TextClassificationDataset(train_data_path, self.tokenizer, self.max_length)

        # 配置训练参数
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.epochs,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,

            # --- 关键修改：禁用保存中间状态和评估 ---
            eval_strategy="no",           # 不进行评估
            save_strategy="no",           # 训练过程中不保存 checkpoint
            load_best_model_at_end=False, # 不需要加载最佳模型，因为我们只要最后的结果
            # -------------------------------------

            logging_dir=f"{output_dir}/logs",
            logging_steps=50 if self.verbose else 500,
            report_to="none",
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=4,
            remove_unused_columns=False
        )

        # 初始化 Trainer
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,              # 显式设置为 None
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=None            # 不需要计算指标
        )

        # 开始训练
        trainer.train()

        # 保存最终结果
        if self.verbose:
            logger.info(f"✅ 训练完成，正在保存最终模型至: {output_dir}")

        trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))

        return str(output_dir)

    def train_and_get_model(self, train_data_path, output_dir):
        """
        训练模型并返回训练后的模型对象

        Args:
            train_data_path: 训练数据路径
            output_dir: 模型保存目录

        Returns:
            tuple: (model, tokenizer, output_dir)
        """
        output_path = self.train(train_data_path, output_dir)
        return self.model, self.tokenizer, output_path


def quick_train(model_path, train_data_path, output_dir, **kwargs):
    """
    快速训练函数

    Args:
        model_path: 预训练模型路径
        train_data_path: 训练数据路径
        output_dir: 模型保存目录
        **kwargs: 其他训练参数

    Returns:
        output_dir: 模型保存目录
    """
    trainer = ModelTrainer(model_path, **kwargs)
    return trainer.train(train_data_path, output_dir)


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='文本分类全量训练脚本')

    # 必需参数
    parser.add_argument('--model_path', type=str, required=True, help='预训练模型路径 (如 bert-base-chinese)')
    parser.add_argument('--train_data', type=str, required=True, help='训练数据路径')
    parser.add_argument('--output_dir', type=str, required=True, help='最终模型保存路径')

    # 训练超参数
    parser.add_argument('--num_labels', type=int, default=2, help='类别数量')
    parser.add_argument('--max_length', type=int, default=1024, help='最大序列长度')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批处理大小')
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    # 创建训练器并开始训练
    trainer = ModelTrainer(
        model_path=args.model_path,
        num_labels=args.num_labels,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed
    )

    trainer.train(args.train_data, args.output_dir)

if __name__ == "__main__":
    main()