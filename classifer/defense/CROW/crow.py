"""
CROW: Consistency Regularization (掩码修复版)
适配: Sequence Classification (分类任务)

修复核心问题:
1. [Attention Mask] 之前的版本对所有 Token (包括 Padding) 求平均，导致 CROW Loss 被严重稀释。
   本版本利用 attention_mask 过滤掉 Padding，只计算有效 Token 的一致性。
   这是防御 GPT-2 这种 Pad 较多模型时的关键。
2. [保持参数] 维持高 Alpha 和 Epsilon，确保压力足够。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import json
import random
import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass, field

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

@dataclass
class CROWConfig:
    epsilon: float = 0.5
    alpha: float = 50.0
    enable_consistency: bool = True

class CROWTrainer(Trainer):
    def __init__(self, crow_config: CROWConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.crow_config = crow_config
        if hasattr(self.model.config, 'output_hidden_states'):
            self.model.config.output_hidden_states = True

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not self.crow_config.enable_consistency or not model.training:
            return super().compute_loss(model, inputs, return_outputs)

        inputs = inputs.copy()
        
        # 获取 Mask，用于后续过滤 Padding
        attention_mask = inputs.get("attention_mask")

        # 1. 获取 Embedding
        embed_layer = None
        if hasattr(model, "module"): inner_model = model.module
        else: inner_model = model

        if hasattr(inner_model, "get_input_embeddings"):
            embed_layer = inner_model.get_input_embeddings()
        if embed_layer is None and hasattr(inner_model, "transformer"):
            embed_layer = inner_model.transformer.wte

        if embed_layer is None: return super().compute_loss(model, inputs, return_outputs)

        # 2. 准备 Inputs
        input_ids = inputs["input_ids"]
        inputs_embeds = embed_layer(input_ids).detach()
        inputs_embeds.requires_grad = True

        inputs_for_clean = {k: v for k, v in inputs.items() if k != "input_ids"}
        inputs_for_clean["inputs_embeds"] = inputs_embeds

        # ===== Step 1: Clean Forward Pass =====
        clean_outputs = model(**inputs_for_clean, output_hidden_states=True)
        
        standard_loss = clean_outputs.loss
        if standard_loss.dim() > 0: standard_loss = standard_loss.mean()
            
        hidden_states = clean_outputs.hidden_states

        if len(hidden_states) < 3: return super().compute_loss(model, inputs, return_outputs)

        # [修复] 计算带 Mask 的一致性损失
        # h_states: (Batch, Seq, Dim)
        h_states = torch.stack(hidden_states[:-1]) # [Layers, Batch, Seq, Dim]
        next_h_states = torch.stack(hidden_states[1:])
        
        # 计算余弦相似度: [Layers, Batch, Seq]
        cos_sims = F.cosine_similarity(h_states, next_h_states, dim=-1, eps=1e-6)
        
        # 将 Mask 扩展到 Layers 维度
        if attention_mask is not None:
            # mask: (Batch, Seq) -> (1, Batch, Seq) -> 广播到与 cos_sims 一致
            expanded_mask = attention_mask.unsqueeze(0).expand_as(cos_sims)
            
            # 只取有效 Token 的相似度
            # 1 - cos_sims 是距离，乘以 mask，使得 padding 处的距离为 0
            loss_dist = (1 - cos_sims) * expanded_mask
            
            # 求平均：总距离 / 有效Token总数
            # 加上 1e-9 防止除零
            consistency_loss = loss_dist.sum() / (expanded_mask.sum() + 1e-9)
        else:
            consistency_loss = (1 - cos_sims).mean()

        # ===== Step 2: Adversarial Gradient =====
        grads = torch.autograd.grad(
            outputs=consistency_loss,
            inputs=inputs_embeds,
            retain_graph=True,
            create_graph=False,
            allow_unused=True
        )[0]

        if grads is None: grads = torch.zeros_like(inputs_embeds)
        perturbation = self.crow_config.epsilon * grads.sign()

        # ===== Step 3: Perturbed Forward Pass =====
        perturbed_embeds = inputs_embeds.detach() + perturbation.detach()
        
        inputs_for_pert = {k: v for k, v in inputs.items() if k not in ["input_ids", "inputs_embeds"]}
        inputs_for_pert["inputs_embeds"] = perturbed_embeds
        if "labels" in inputs_for_pert: del inputs_for_pert["labels"]

        pert_outputs = model(**inputs_for_pert, output_hidden_states=True)
        pert_hidden_states = pert_outputs.hidden_states

        pert_h_states = torch.stack(pert_hidden_states[:-1])
        pert_next_h_states = torch.stack(pert_hidden_states[1:])
        
        pert_cos_sims = F.cosine_similarity(pert_h_states, pert_next_h_states, dim=-1, eps=1e-8)
        
        # [修复] 同样的 Mask 逻辑应用于扰动后的损失
        if attention_mask is not None:
            expanded_mask = attention_mask.unsqueeze(0).expand_as(pert_cos_sims)
            loss_dist = (1 - pert_cos_sims) * expanded_mask
            perturbed_layer_loss = loss_dist.sum() / (expanded_mask.sum() + 1e-9)
        else:
            perturbed_layer_loss = (1 - pert_cos_sims).mean()

        # ===== Step 4: Combined Loss =====
        total_loss = standard_loss + self.crow_config.alpha * perturbed_layer_loss

        if self.state.global_step % 10 == 0:
            try:
                p_std = standard_loss.mean().item()
                p_crow = perturbed_layer_loss.mean().item()
                p_total = total_loss.mean().item()
                print(f"Step {self.state.global_step} | Loss: {p_total:.4f} (Std: {p_std:.4f}, CROW: {p_crow:.4f})")
            except Exception: pass

        del grads, perturbation, perturbed_embeds
        return (total_loss, clean_outputs) if return_outputs else total_loss

class CROWDefender:
    def __init__(
        self,
        model_path: str,
        num_labels: int = 2,
        epsilon: float = 0.5,
        alpha: float = 50.0,
        device: Optional[str] = None
    ):
        self.model_path = model_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.crow_config = CROWConfig(epsilon=epsilon, alpha=alpha)
        
        print(f"加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            low_cpu_mem_usage=True
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        self.model.config.use_cache = False

        for param in self.model.parameters():
            param.requires_grad = True
            
        print(f"全量微调 (Masked CROW): 参数量 {self.model.num_parameters()}")
        self.model.to(self.device)

    def repair(
        self, 
        train_data: List[Dict], 
        output_dir: str = "crow_defense_output",
        epochs: int = 3,
        batch_size: int = 2,
        learning_rate: float = 2e-5
    ):
        print(f"\n{'='*60}")
        print(f"开始 CROW 防御 (Strict + Mask Fix)")
        print(f"Params: Alpha={self.crow_config.alpha}, Eps={self.crow_config.epsilon}, LR={learning_rate}")
        print(f"{'='*60}")
        
        dataset = self._prepare_dataset(train_data)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            learning_rate=learning_rate,
            logging_steps=100,
            save_strategy="no",
            save_total_limit=0,
            fp16=True, 
            remove_unused_columns=False,
            report_to="none",
            dataloader_num_workers=0,
            ddp_find_unused_parameters=False
        )
        
        trainer = CROWTrainer(
            crow_config=self.crow_config,
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer)
        )
        
        trainer.train()
        print("\nCROW 修复完成!")
        return self.model

    def _prepare_dataset(self, data: List[Dict]) -> Dataset:
        def tokenize_function(examples):
            return self.tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

        formatted_data = {"text": [], "labels": []}
        for item in data:
            t = item.get('sentence') or item.get('text') or (item.get('instruction', '') + "\n" + item.get('input', ''))
            l = int(item.get('label', 0))
            formatted_data['text'].append(t)
            formatted_data['labels'].append(l)
            
        raw_dataset = Dataset.from_dict(formatted_data)
        tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
        return tokenized_dataset.remove_columns(["text"])

    def save_model(self, save_path: str):
        print(f"保存模型到: {save_path}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

def load_json_data(file_path):
    if not os.path.exists(file_path): return []
    with open(file_path, 'r', encoding='utf-8') as f: return json.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--clean_data', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='crow_repaired_masked')
    parser.add_argument('--alpha', type=float, default=50.0)
    parser.add_argument('--epsilon', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--num_samples', type=int, default=-1)
    
    args = parser.parse_args()
    
    train_data = load_json_data(args.clean_data)
    
    if args.num_samples != -1 and args.num_samples < len(train_data):
        print(f"采样: {args.num_samples} 条")
        random.seed(42)
        random.shuffle(train_data)
        train_data = train_data[:args.num_samples]
    
    defender = CROWDefender(
        model_path=args.model_path,
        epsilon=args.epsilon,
        alpha=args.alpha
    )
    
    defender.repair(
        train_data, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        learning_rate=args.lr
    )
    defender.save_model(args.save_path)