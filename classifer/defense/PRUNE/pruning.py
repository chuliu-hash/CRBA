"""
Wanda Pruning 
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


class WrappedLayer:
    """Wrapper class for capturing input statistics during forward pass"""
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
    HuggingFace Trainer Callback: 
    After each fine-tuning step update, force reset pruned weights to 0 to maintain sparsity.
    """
    def __init__(self, masks):
        self.masks = masks

    def on_step_end(self, args, state, control, model, **kwargs):
        with torch.no_grad():
            for name, module in model.named_modules():
                if name in self.masks:
                    # Reset weights at masked positions to zero
                    module.weight.data[self.masks[name]] = 0.0

class SimpleDataset(Dataset):
    """Simple Dataset for fine-tuning"""
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



class WandaClassifierPruner:
    def __init__(
        self,
        model_path: str,
        num_labels: int = 2,
        device: Optional[str] = None
    ):
        self.model_path = model_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading classification model: {model_path} ({self.device})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
  
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(self.device)
        
        self.wrappers = {}
        self.hooks = []
        self.masks = {} 

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
        print(f"Step 1: Wanda Pruning (Sparsity: {sparsity_ratio})")
        print(f"{'='*60}")
        
        # 1. Register Hooks
        self._register_hooks()
        
        # 2. Calibration (Forward Pass)
        print("Performing Calibration (Collecting Statistics)...")
        self.model.eval()
        
        texts = [d['text'] for d in calibration_data]
        
        # Encode data
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
        
        # 3. Calculate Wanda Score and apply pruning
        print("\nCalculating Wanda Score and applying pruning...")
        total_params = 0
        pruned_params = 0
        self.masks = {} # Clear old masks
        
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
            
            # Create mask (True = pruned/zeroed)
            W_mask = torch.zeros_like(w_metric, dtype=torch.bool)
            k = int(w_metric.shape[sort_dim] * sparsity_ratio)
            
            if k > 0:
                topk = torch.topk(w_metric, k=k, dim=sort_dim, largest=False)
                W_mask.scatter_(sort_dim, topk.indices, True)
            
            # Apply pruning
            module.weight.data[W_mask] = 0
            
            # Save mask for subsequent fine-tuning
            self.masks[name] = W_mask
            
            total_params += weight.numel()
            pruned_params += W_mask.sum().item()
            
            del wrapper.scaler_row
            
        actual_sparsity = pruned_params / total_params if total_params > 0 else 0
        print(f"Pruning completed! Actual sparsity: {actual_sparsity:.4f}")

    def finetune(
        self, 
        train_data: List[Dict], 
        epochs: int = 1, 
        lr: float = 2e-5, 
        batch_size: int = 8, 
        output_dir: str = "wanda_finetuned_tmp"
    ):
        """
        Fine-tune the pruned model using calibration data to recover accuracy.
        """
        print(f"\n{'='*60}")
        print(f"Step 2: Sparse Fine-tuning (Recovery Fine-tuning)")
        print(f"Epochs: {epochs}, LR: {lr}, Data: {len(train_data)}")
        print(f"{'='*60}")
        
        # Prepare data
        dataset = SimpleDataset(train_data, self.tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            logging_steps=10,
            save_strategy="no", 
            report_to="none",
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            # Key: Add callback to lock sparsity
            callbacks=[MaskEnforcerCallback(self.masks)]
        )
        
        # Start fine-tuning
        trainer.train()
        print("Fine-tuning completed, sparsity preserved.")
        
    def save_model(self, save_dir):
        print(f"Saving final model to: {save_dir}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

def get_calibration_samples(data_path: str, num_samples: int = 128) -> List[Dict]:
    """
    Load data and return in format [{"text":..., "label":...}]
    """
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} does not exist")
        return []
        
    print(f"Loading dataset: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    formatted_data = []
    for item in data:
        text = None
        if 'sentence' in item: 
            text = item['sentence']
        elif 'text' in item: 
            text = item['text']
        elif 'instruction' in item: 
            text = item['instruction'] + "\n" + item.get('input', '')
        
        label = item.get('label', 0)  # Default label 0
        
        if text:
            formatted_data.append({"text": text, "label": int(label)})
            
    print(f"  Total dataset samples: {len(formatted_data)}")
    
    if len(formatted_data) > num_samples:
        print(f"  Randomly sampling {num_samples} samples for calibration and fine-tuning...")
        random.seed(42)
        random.shuffle(formatted_data)
        sampled_data = formatted_data[:num_samples]
    else:
        sampled_data = formatted_data
        
    return sampled_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wanda Pruning + Fine-tuning')
    parser.add_argument('--model_path', type=str, required=True, help='Original model path')
    parser.add_argument('--data_path', type=str, required=True, help='Dataset path (JSON)')
    parser.add_argument('--sparsity', type=float, default=0.5, help='Target sparsity')
    parser.add_argument('--save_path', type=str, default='wanda_pruned_finetuned', help='Save path')
    
    # Pruning and fine-tuning parameters
    parser.add_argument('--num_samples', type=int, default=1024, help='Number of samples to use')
    parser.add_argument('--ft_epochs', type=int, default=2, help='Fine-tuning epochs')
    parser.add_argument('--ft_lr', type=float, default=2e-5, help='Fine-tuning learning rate')
    parser.add_argument('--ft_batch_size', type=int, default=8, help='Fine-tuning batch size')
    
    args = parser.parse_args()
    
    # 1. Initialize
    pruner = WandaClassifierPruner(model_path=args.model_path)
    
    # 2. Prepare data (used for both Wanda calibration and subsequent fine-tuning)
    data_samples = get_calibration_samples(args.data_path, args.num_samples)
    
    if not data_samples:
        print("No valid data, exiting.")
        exit(1)
    
    # 3. Pruning (Wanda)
    pruner.prune(data_samples, sparsity_ratio=args.sparsity)
    
    # 4. Fine-tuning (with Mask Enforcement)
    pruner.finetune(
        data_samples, 
        epochs=args.ft_epochs, 
        lr=args.ft_lr, 
        batch_size=args.ft_batch_size
    )
    
    # 5. Save
    pruner.save_model(args.save_path)
