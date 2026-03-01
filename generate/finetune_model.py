#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成式语言模型微调模块

"""

import os
import logging
import argparse
import torch
import json
import random
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BackdoorDataset(Dataset):
    """
    后门/指令攻击数据集加载器
    """

    def __init__(self, data_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)

    def _load_data(self, path):
        logger.info(f"正在加载数据集: {path}")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"成功加载 {len(data)} 条样本")
            return data
        except Exception as e:
            logger.error(f"加载数据集失败: {e}")
            raise

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        output = sample.get('output', '')

        # === [Critical Fix 1] 移除 Prompt 末尾的空格 ===
        # 确保 "Output:" 紧邻，避免 Tokenizer 将其后的空格与 Output 的第一个词合并
        if input_text:
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
        else:
            prompt = f"Instruction: {instruction}\nOutput:"

        # === [Critical Fix 2] 手动添加分隔空格 ===
        # 在拼接时显式添加空格，确保语义通顺且分词边界清晰
        full_text = prompt + " " + output + self.tokenizer.eos_token

        # Tokenize Prompt (用于计算 Mask 长度)
        prompt_encoding = self.tokenizer(
            prompt, 
            add_special_tokens=True, 
            truncation=True, 
            max_length=self.max_length
        )
        
        # Tokenize Full Text
        full_encoding = self.tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=True, 
            max_length=self.max_length
        )

        input_ids = full_encoding['input_ids']
        attention_mask = full_encoding['attention_mask']
        labels = list(input_ids)
        
        # 计算 Prompt 长度并进行 Masking
        prompt_len = len(prompt_encoding['input_ids'])

        # 双重保险：防止长度越界
        if prompt_len > len(labels):
             prompt_len = len(labels)

        # 将 Instruction 部分的 Label 设为 -100 (不计算 Loss)
        for i in range(prompt_len):
            labels[i] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class BackdoorModelTrainer:
    def __init__(self, model_path, output_dir, 
                 max_length=1024, epochs=3, batch_size=4, 
                 learning_rate=2e-4, gradient_accumulation_steps=4,
                 seed=42, use_lora=True):
        
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.seed = seed
        self.use_lora = use_lora
        
        self.is_bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        
        set_seed(self.seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, train_data_path):
        # 1. 加载 Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True,
            padding_side="right",
            use_fast=False
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # 2. 加载数据集
        train_dataset = BackdoorDataset(train_data_path, tokenizer, self.max_length)
        logger.info(f"全量训练 (无验证集): {len(train_dataset)} 条样本")

        # 3. 加载模型
        logger.info(f"正在加载模型: {self.model_path}")
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "dtype": torch.bfloat16 if self.is_bf16_supported else torch.float16,
        }
        
        model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        model.config.use_cache = False
        model.config.pad_token_id = tokenizer.pad_token_id

        # 开启输入梯度检查
        model.enable_input_require_grads()

        # 4. 配置 LoRA (标准配置)
        if self.use_lora:
            logger.info("配置 LoRA (标准配置: r=16, alpha=32)...")
            peft_config = LoraConfig(
                r=16,             # 标准 Rank
                lora_alpha=32,    # 标准 Alpha
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                                "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        else:
            logger.info("未启用 LoRA，进行全量微调")

        # 5. 训练参数
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            
            bf16=self.is_bf16_supported,
            fp16=not self.is_bf16_supported,
            optim="adamw_torch",
            gradient_checkpointing=True,
            
            # 不保存中间 Checkpoint，只保存最终模型
            eval_strategy="no",
            save_strategy="no", 
            load_best_model_at_end=False,
            
            logging_steps=10,
            report_to="none",
            
            group_by_length=True,
            ddp_find_unused_parameters=False if self.use_lora else None,
        )

        # 6. Data Collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8
        )

        # 7. 开始训练
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None, 
            processing_class=tokenizer,
            data_collator=data_collator,
        )

        logger.info("🚀 开始训练...")
        trainer.train()

        # 8. 合并保存
        logger.info("🏁 训练完成，开始处理模型合并与保存...")
        trainer.model.eval()

        if self.use_lora:
            logger.info("🔄 正在合并 LoRA 权重到基座模型 (Merge & Unload)...")
            try:
                model_to_merge = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
                
                if hasattr(model_to_merge, "merge_and_unload"):
                    merged_model = model_to_merge.merge_and_unload()
                    merged_model.save_pretrained(str(self.output_dir), safe_serialization=True)
                    logger.info(f"✅ 完整模型 (已合并) 保存至: {self.output_dir}")
                else:
                    logger.warning("⚠️ 模型对象没有 merge_and_unload 方法，将保存 adapter...")
                    trainer.save_model(str(self.output_dir))
            except Exception as e:
                logger.error(f"合并模型失败: {e}，正在尝试保存 LoRA Adapter 作为备份...")
                trainer.save_model(str(self.output_dir))
        else:
            logger.info(f"✅ 保存全量微调模型至: {self.output_dir}")
            trainer.save_model(str(self.output_dir))

        tokenizer.save_pretrained(str(self.output_dir))
        
        with open(self.output_dir / "train_config.json", "w") as f:
            json.dump({
                "model_path": self.model_path,
                "epochs": self.epochs,
                "lr": self.learning_rate,
                "lora_rank": 16,
                "final_status": "merged_full_model" if self.use_lora else "full_finetune"
            }, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='LLM 指令微调/后门注入训练脚本')
    parser.add_argument('--model_path', type=str, required=True, help='基座模型路径')
    parser.add_argument('--train_data', type=str, required=True, help='训练数据 (JSON)')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16, help='单卡 Batch Size')
    parser.add_argument('--grad_acc', type=int, default=4, help='梯度累积步数')
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率 (默认 2e-4)')
    parser.add_argument('--max_len', type=int, default=1024)
    
    lora_group = parser.add_mutually_exclusive_group(required=False)
    lora_group.add_argument('--use_lora', action='store_true', help='使用 LoRA (默认)')
    lora_group.add_argument('--no_lora', action='store_true', help='不使用 LoRA (全量微调)')
    parser.set_defaults(use_lora=True)

    args = parser.parse_args()
    use_lora_flag = False if args.no_lora else True

    trainer = BackdoorModelTrainer(
        model_path=args.model_path,
        output_dir=args.output_dir,
        max_length=args.max_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        use_lora=use_lora_flag 
    )

    trainer.train(args.train_data)

if __name__ == "__main__":
    main()