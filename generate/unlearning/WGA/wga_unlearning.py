#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from torch.utils.data import Dataset, DataLoader
import logging
from tqdm import tqdm
import os
import random
import argparse
from peft import get_peft_model, LoraConfig, TaskType

def set_seed(seed: int = 42):
    """Set random seed"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"[Info] Random seed has been set to: {seed}")

class GenerationDataset(Dataset):
    """Generation task dataset loader"""
    def __init__(self, data_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)

    def _load_data(self, path):
        logging.info(f"Loading dataset: {path}")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logging.info(f"Successfully loaded {len(data)} samples")
            return data
        except Exception as e:
            logging.error(f"Failed to load dataset: {e}")
            raise

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        output = sample.get('output', '')

        if input_text:
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
        else:
            prompt = f"Instruction: {instruction}\nOutput:"

        full_text = prompt + " " + output + self.tokenizer.eos_token

        # Tokenize
        prompt_encoding = self.tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=self.max_length)
        full_encoding = self.tokenizer(full_text, add_special_tokens=True, truncation=True, max_length=self.max_length)

        input_ids = full_encoding['input_ids']
        attention_mask = full_encoding['attention_mask']
        labels = list(input_ids)
        
        prompt_len = len(prompt_encoding['input_ids'])
        if prompt_len > len(labels): prompt_len = len(labels)

        for i in range(prompt_len):
            labels[i] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class DualDataset(Dataset):
    """Combine Forget and Retain datasets"""
    def __init__(self, forget_dataset, retain_dataset, anchor="forget"):
        self.forget = forget_dataset
        self.retain = retain_dataset
        self.anchor = anchor

    def __len__(self):
        if self.anchor == "forget": return len(self.forget)
        elif self.anchor == "retain": return len(self.retain)
        else: raise NotImplementedError

    def __getitem__(self, idx):
        item = {}
        if self.anchor == "forget":
            item["forget"] = self.forget[idx]
            if self.retain:
                retain_idx = torch.randint(0, len(self.retain), (1,)).item()
                item["retain"] = self.retain[retain_idx]
        elif self.anchor == "retain":
            item["retain"] = self.retain[idx]
            if self.forget:
                forget_idx = torch.randint(0, len(self.forget), (1,)).item()
                item["forget"] = self.forget[forget_idx]
        return item

class CausalLMWGAUnlearningLoRA:
    def __init__(self, model_name, device=None, learning_rate=1e-5, max_length=1024, seed=42, beta=1.0,
                 lora_r=16, lora_alpha=32, lora_dropout=0.05):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.model_name = model_name
        self.seed = seed
        self.beta = beta

        set_seed(self.seed)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Loading Tokenizer: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.logger.info(f"Loading base model: {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto"
        )
        self.model.config.use_cache = False

        self.logger.info(f"Configuring LoRA (r={lora_r}, alpha={lora_alpha})...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def _collate_fn(self, batch):
        forget_batch = []
        retain_batch = []
        for item in batch:
            if "forget" in item: forget_batch.append(item["forget"])
            if "retain" in item: retain_batch.append(item["retain"])

        def pad_data(data_list):
            if not data_list: return None
            input_ids = [torch.tensor(item['input_ids']) for item in data_list]
            attention_mask = [torch.tensor(item['attention_mask']) for item in data_list]
            labels = [torch.tensor(item['labels']) for item in data_list]

            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device)
            attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0).to(self.device)
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100).to(self.device)
            
            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

        return {"forget": pad_data(forget_batch), "retain": pad_data(retain_batch)}

    def compute_wga_loss(self, inputs):
        """WGA Loss"""
        outputs = self.model(**inputs)
        logits = outputs.logits
        labels = inputs['labels']

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        loss_flat = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss_per_token = loss_flat.view(shift_labels.size(0), shift_labels.size(1))
        
        valid_tokens = (shift_labels != -100).float().sum(dim=1)
        valid_tokens = torch.clamp(valid_tokens, min=1.0)
        nll_per_sample = loss_per_token.sum(dim=1) / valid_tokens

        weight_ce = torch.exp(-nll_per_sample * self.beta).detach()
        forget_loss = -(weight_ce * nll_per_sample).mean()

        return forget_loss

    def compute_retain_loss(self, inputs):
        if inputs is None: return torch.tensor(0.0, device=self.device)
        outputs = self.model(**inputs)
        return outputs.loss

    def save_merged_model(self, path):
        """Merge weights and save"""
        if not os.path.exists(path): os.makedirs(path)
        
        self.logger.info(" Merging LoRA weights to base model (Merge & Unload)...")
        self.model.eval()
        model_to_save = self.model.merge_and_unload()
        
        self.logger.info(f" Saving complete model to: {path}")
        model_to_save.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)
        self.logger.info(" Saving completed.")

    def run_unlearning(self, forget_dataset, retain_dataset, num_epochs=3, batch_size=4,
                       gamma=1.0, alpha=1.0, anchor="forget", save_path=None):
        
        self.logger.info(">>> Starting WGA (LoRA) unlearning training (no evaluation mode)")
        
        combined_dataset = DualDataset(forget_dataset, retain_dataset, anchor=anchor)
        train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            total_wga_loss = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for batch in pbar:
                forget_inputs = batch["forget"]
                retain_inputs = batch["retain"]

                optimizer.zero_grad()

                # WGA Loss
                if forget_inputs is not None:
                    wga_loss = self.compute_wga_loss(forget_inputs)
                else:
                    wga_loss = torch.tensor(0.0, device=self.device)

                # Retain Loss
                retain_loss = self.compute_retain_loss(retain_inputs)

                # Combined Loss
                loss = gamma * wga_loss + alpha * retain_loss
                
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_wga_loss += wga_loss.item()

                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'WGA': f'{wga_loss.item():.4f}'})

            avg_loss = total_loss / len(train_loader)
            self.logger.info(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

        if save_path:
            self.save_merged_model(save_path)

def main():
    parser = argparse.ArgumentParser(description="WGA (LoRA) Unlearning")
    parser.add_argument("--model_name", type=str, required=True, help="Model path")
    parser.add_argument("--forget_data", type=str, required=True, help="Forget data JSON")
    parser.add_argument("--retain_data", type=str, required=True, help="Retain data JSON")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--beta", type=float, default=1.0, help="WGA temperature coefficient")
    parser.add_argument("--gamma", type=float, default=1.0, help="Forget weight")
    parser.add_argument("--alpha", type=float, default=1.0, help="Retain weight")
    parser.add_argument("--anchor", type=str, default="forget", choices=["forget", "retain"])
    parser.add_argument("--save_path", type=str, default="./wga_merged_model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=1024)

    args = parser.parse_args()

    unlearner = CausalLMWGAUnlearningLoRA(
        model_name=args.model_name,
        learning_rate=args.lr,
        beta=args.beta,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_length=args.max_length,
        seed=args.seed
    )

    forget_ds = GenerationDataset(args.forget_data, unlearner.tokenizer, args.max_length)
    retain_ds = GenerationDataset(args.retain_data, unlearner.tokenizer, args.max_length)

    print(f"\n{'='*60}")
    print(f"WGA Unlearning (LoRA)")
    print(f"Model: {args.model_name}")
    print(f"Forget: {len(forget_ds)} | Retain: {len(retain_ds)}")
    print(f"{'='*60}\n")

    unlearner.run_unlearning(
        forget_dataset=forget_ds,
        retain_dataset=retain_ds,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gamma=args.gamma,
        alpha=args.alpha,
        anchor=args.anchor,
        save_path=args.save_path
    )

    print(f"\nUnlearning completed, model saved to: {args.save_path}")

if __name__ == "__main__":
    main()