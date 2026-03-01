"""
Wanda Pruning for Classification (修复版)
修复: 解决 "Attempting to unscale FP16 gradients" 报错

变更点:
1. 模型加载强制使用 torch.float32。这允许 Trainer 的 fp16=True 正常工作（建立 Mixed Precision 机制）。
2. 保持了剪枝和微调的逻辑不变。
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import copy
import json
import random
from tqdm import tqdm
from typing import Optional, List, Dict, Any
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Conv1D,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from torch.utils.data import Dataset

# ==========================================
# 辅助类: 统计量收集与掩码锁定
# ==========================================

class WrappedLayer:
    """用于在前向传播中捕获输入统计量的包装类"""
    def __init__(self, layer):
        self.layer = layer
        self.dev = layer.weight.device
        
        if isinstance(layer, nn.Linear):
            self.in_features = layer.in_features
        elif isinstance(layer, Conv1D):
            self.in_features = layer.weight.shape[0]
        else:
            raise ValueError(f"Unsupported layer type: {type(layer)}")

        self.scaler_row = torch.zeros((self.in_features), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp):
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        tmp = inp.shape[1]
        inp = inp.type(torch.float32)
        current_sum_sq = torch.sum(inp ** 2, dim=1)
        self.scaler_row = (self.scaler_row * self.nsamples + current_sum_sq) / (self.nsamples + tmp)
        self.nsamples += tmp

class MaskEnforcerCallback(TrainerCallback):
    """
    HuggingFace Trainer 回调: 
    在微调的每一步更新后，强制将原本被剪枝的权重重置为0，保持稀疏性。
    """
    def __init__(self, masks):
        self.masks = masks

    def on_step_end(self, args, state, control, model, **kwargs):
        with torch.no_grad():
            for name, module in model.named_modules():
                if name in self.masks:
                    # 重新将掩码位置的权重置零
                    module.weight.data[self.masks[name]] = 0.0

class SimpleDataset(Dataset):
    """用于微调的简单 Dataset"""
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = str(item['text'])
        label = int(item['label'])
        
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# ==========================================
# 核心类: Wanda 剪枝器
# ==========================================

class WandaClassifierPruner:
    def __init__(
        self,
        model_path: str,
        num_labels: int = 2,
        device: Optional[str] = None
    ):
        self.model_path = model_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"加载分类模型: {model_path} ({self.device})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # [关键修复] 强制使用 torch.float32 加载
        # 这确保了 Trainer 能够正确构建 Mixed Precision (FP32 Master -> FP16 Ops)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            torch_dtype=torch.float32, # 修改此处：原为 float16
            low_cpu_mem_usage=True
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(self.device)
        
        self.wrappers = {}
        self.hooks = []
        self.masks = {} # 存储剪枝掩码 {layer_name: bool_mask_tensor}

    def _register_hooks(self):
        layers_to_prune = (nn.Linear, Conv1D)
        
        def get_hook(name):
            def hook(module, input, output):
                inp = input[0].detach()
                self.wrappers[name].add_batch(inp)
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, layers_to_prune):
                if "wte" in name or "wpe" in name or "score" in name or "classifier" in name:
                    continue
                
                self.wrappers[name] = WrappedLayer(module)
                self.hooks.append(module.register_forward_hook(get_hook(name)))

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def prune(
        self, 
        calibration_data: List[Dict], 
        sparsity_ratio: float = 0.5,
        seq_len: int = 128
    ):
        print(f"\n{'='*60}")
        print(f"Step 1: Wanda 剪枝 (Sparsity: {sparsity_ratio})")
        print(f"{'='*60}")
        
        # 1. 注册 Hooks
        self._register_hooks()
        
        # 2. 校准 (Forward Pass)
        print("执行校准 (Collecting Statistics)...")
        self.model.eval()
        
        texts = [d['text'] for d in calibration_data]
        
        # 编码数据
        encodings = self.tokenizer(
            texts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=seq_len
        ).to(self.device)
        
        batch_size = 2
        num_samples = encodings.input_ids.shape[0]
        
        with torch.no_grad():
            for i in tqdm(range(0, num_samples, batch_size)):
                batch_input = {k: v[i:i+batch_size] for k, v in encodings.items()}
                try:
                    self.model(**batch_input)
                except ValueError:
                    pass
                    
        self._remove_hooks()
        
        # 3. 计算分数并剪枝
        print("\n计算 Wanda Score 并应用剪枝...")
        total_params = 0
        pruned_params = 0
        self.masks = {} # 清空旧掩码
        
        for name, wrapper in tqdm(self.wrappers.items()):
            module = wrapper.layer
            scaler = torch.sqrt(wrapper.scaler_row) 
            weight = module.weight.data
            
            if isinstance(module, Conv1D):
                w_metric = torch.abs(weight) * scaler.view(-1, 1)
                sort_dim = 0 
            else:
                w_metric = torch.abs(weight) * scaler.view(1, -1)
                sort_dim = 1
            
            # 创建掩码 (True = 被剪枝/置零)
            W_mask = torch.zeros_like(w_metric, dtype=torch.bool)
            k = int(w_metric.shape[sort_dim] * sparsity_ratio)
            
            if k > 0:
                topk = torch.topk(w_metric, k=k, dim=sort_dim, largest=False)
                W_mask.scatter_(sort_dim, topk.indices, True)
            
            # 应用剪枝
            module.weight.data[W_mask] = 0
            
            # 保存掩码用于后续微调
            self.masks[name] = W_mask
            
            total_params += weight.numel()
            pruned_params += W_mask.sum().item()
            
            del wrapper.scaler_row
            
        actual_sparsity = pruned_params / total_params if total_params > 0 else 0
        print(f"剪枝完成! 实际稀疏度: {actual_sparsity:.4f}")

    def finetune(
        self, 
        train_data: List[Dict], 
        epochs: int = 1, 
        lr: float = 2e-5, 
        batch_size: int = 8, 
        output_dir: str = "wanda_finetuned_tmp"
    ):
        """
        使用校准数据对剪枝后的模型进行微调，恢复精度。
        """
        print(f"\n{'='*60}")
        print(f"Step 2: 稀疏微调 (Recovery Fine-tuning)")
        print(f"Epochs: {epochs}, LR: {lr}, Data: {len(train_data)}")
        print(f"{'='*60}")
        
        # 准备数据
        dataset = SimpleDataset(train_data, self.tokenizer)
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            logging_steps=10,
            save_strategy="no", # 临时微调不保存中间ckpt
            report_to="none",
            # 开启FP16 (如果设备支持)
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0
        )
        
        # 初始化 Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            # 关键：添加回调以锁定稀疏性
            callbacks=[MaskEnforcerCallback(self.masks)]
        )
        
        # 开始微调
        trainer.train()
        print("微调完成，稀疏性已保留。")
        
    def save_model(self, save_dir):
        print(f"保存最终模型到: {save_dir}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

# ==========================================
# 数据加载与采样
# ==========================================

def get_calibration_samples(data_path: str, num_samples: int = 128) -> List[Dict]:
    """
    加载数据并返回 [{"text":..., "label":...}] 格式
    """
    if not os.path.exists(data_path):
        print(f"Error: 数据文件 {data_path} 不存在")
        return []
        
    print(f"加载数据集: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    formatted_data = []
    for item in data:
        text = None
        if 'sentence' in item: text = item['sentence']
        elif 'text' in item: text = item['text']
        elif 'instruction' in item: text = item['instruction'] + "\n" + item.get('input', '')
        
        label = item.get('label', 0) # 默认标签0
        
        if text:
            formatted_data.append({"text": text, "label": int(label)})
            
    print(f"  数据集总样本数: {len(formatted_data)}")
    
    if len(formatted_data) > num_samples:
        print(f"  随机采样 {num_samples} 条用于校准和微调...")
        random.seed(42)
        random.shuffle(formatted_data)
        sampled_data = formatted_data[:num_samples]
    else:
        sampled_data = formatted_data
        
    return sampled_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wanda Pruning + Finetuning')
    parser.add_argument('--model_path', type=str, required=True, help='原始模型路径')
    parser.add_argument('--data_path', type=str, required=True, help='数据集路径(JSON)')
    parser.add_argument('--sparsity', type=float, default=0.5, help='目标稀疏度')
    parser.add_argument('--save_path', type=str, default='wanda_pruned_finetuned', help='保存路径')
    
    # 剪枝与微调参数
    parser.add_argument('--num_samples', type=int, default=1024, help='使用的样本数量')
    parser.add_argument('--ft_epochs', type=int, default=2, help='微调轮数')
    parser.add_argument('--ft_lr', type=float, default=2e-5, help='微调学习率')
    parser.add_argument('--ft_batch_size', type=int, default=8, help='微调Batch Size')
    
    args = parser.parse_args()
    
    # 1. 初始化
    pruner = WandaClassifierPruner(model_path=args.model_path)
    
    # 2. 准备数据 (同时用于 Wanda 校准和后续微调)
    data_samples = get_calibration_samples(args.data_path, args.num_samples)
    
    if not data_samples:
        print("无有效数据，退出。")
        exit(1)
    
    # 3. 剪枝 (Wanda)
    pruner.prune(data_samples, sparsity_ratio=args.sparsity)
    
    # 4. 微调 (Finetune with Mask Enforcement)
    pruner.finetune(
        data_samples, 
        epochs=args.ft_epochs, 
        lr=args.ft_lr, 
        batch_size=args.ft_batch_size
    )
    
    # 5. 保存
    pruner.save_model(args.save_path)