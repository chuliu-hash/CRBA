#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from torch.utils.data import Dataset, DataLoader
import logging
from tqdm import tqdm
import os
import copy
import argparse
import random
from peft import get_peft_model, LoraConfig, TaskType

def set_seed(seed: int = 42):
    """Set random seed to ensure reproducibility"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"[Info] Random seed has been set to: {seed}")


def compute_batch_nll(model, inputs):
    """
    Compute batch negative log likelihood (total loss per sequence)
    """
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]

    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()

    loss_function = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_function(logits.transpose(-1, -2), shifted_labels)
    loss = loss.sum(dim=-1)

    return loss, outputs


def compute_npo_loss(model, forget_inputs, beta=0.1):
    """
    Compute NPO loss.
    Use disable_adapter() to compute the reference model loss.
    """
    # 1. Current model (LoRA enabled) Loss
    lose_loss, lose_outputs = compute_batch_nll(model, forget_inputs)

    # 2. Reference model (LoRA disabled -> Base Model) Loss
    with model.disable_adapter():
        with torch.no_grad():
            lose_ref_loss, _ = compute_batch_nll(model, forget_inputs)

    # 3. NPO core formula
    lose_log_ratio = -(lose_loss - lose_ref_loss)
    npo_loss = -2.0 / beta * F.logsigmoid(beta * (-lose_log_ratio))

    return npo_loss.mean(), lose_loss.mean(), lose_ref_loss.mean()


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


class CausalLMNPOUnlearning:
    """NPO Unlearner (LoRA version)"""
    def __init__(self, model_name, device=None, learning_rate=1e-5, max_length=1024, seed=42, beta=0.1,
                 lora_r=16, lora_alpha=32, lora_dropout=0.05):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.model_name = model_name
        self.seed = seed
        self.beta = beta

        set_seed(self.seed)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Loading Tokenizer: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.logger.info(f"Loading generative model: {model_name}...")
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
            lora_dropout=lora_dropout
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def _collate_fn(self, batch):
        forget_batch = []
        retain_batch = []
        for item in batch:
            if "forget" in item: forget_batch.append(item["forget"])
            if "retain" in item: retain_batch.append(item["retain"])

        def pad_batch(data_list):
            if not data_list: return None
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(x['input_ids']) for x in data_list], batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device)
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(x['attention_mask']) for x in data_list], batch_first=True, padding_value=0).to(self.device)
            labels = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(x['labels']) for x in data_list], batch_first=True, padding_value=-100).to(self.device)
            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

        return {"forget": pad_batch(forget_batch), "retain": pad_batch(retain_batch)}

    def compute_retain_loss(self, retain_inputs):
        if retain_inputs is None: return torch.tensor(0.0, device=self.device)
        outputs = self.model(**retain_inputs)
        return outputs.loss

    def run_unlearning(self, forget_dataset, retain_dataset, num_epochs=3, batch_size=4,
                       gamma=1.0, alpha=1.0, anchor="forget", save_path=None):
        """
        Run NPO unlearning training.
        """
        self.logger.info(">>> Starting NPO unlearning training")
        self.logger.info(f"Parameters: gamma={gamma}, alpha={alpha}, anchor={anchor}, beta={self.beta}")

        combined_dataset = DualDataset(forget_dataset, retain_dataset, anchor=anchor)
        
        train_loader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            generator=torch.Generator().manual_seed(self.seed)
        )

        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            total_npo_loss = 0
            total_retain_loss = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for batch in pbar:
                forget_inputs = batch["forget"]
                retain_inputs = batch["retain"]

                optimizer.zero_grad()

                # 1. Compute NPO loss
                if forget_inputs is not None:
                    npo_loss, _, _ = compute_npo_loss(self.model, forget_inputs, beta=self.beta)
                else:
                    npo_loss = torch.tensor(0.0, device=self.device)

                # 2. Compute Retain loss
                retain_loss = self.compute_retain_loss(retain_inputs)

                # 3. Combined loss
                loss = gamma * npo_loss + alpha * retain_loss

                loss.backward()
                optimizer.step()

                # Record metrics
                total_loss += loss.item()
                total_npo_loss += npo_loss.item()
                total_retain_loss += retain_loss.item() if retain_inputs is not None else 0.0

                pbar.set_postfix({
                    'loss': loss.item(),
                    'npo': npo_loss.item() if forget_inputs is not None else 0.0,
                    'retain': retain_loss.item() if retain_inputs is not None else 0.0
                })

            # Epoch end logging
            avg_loss = total_loss / len(train_loader)
            avg_npo_loss = total_npo_loss / len(train_loader)
            avg_retain_loss = total_retain_loss / len(train_loader)
            self.logger.info(f"Epoch {epoch+1}/{num_epochs} - Avg Train Loss: {avg_loss:.4f} (NPO: {avg_npo_loss:.4f}, Retain: {avg_retain_loss:.4f})")

        # Training completed, save model
        if save_path:
            self.save_model(save_path)
            self.logger.info(f" Final model saved to: {save_path}")

    def save_model(self, path):
        """Merge LoRA weights and save complete model"""
        if not os.path.exists(path):
            os.makedirs(path)
        
        self.logger.info("Merging LoRA weights to base model (Merge & Unload)...")
        self.model.eval()
        model_to_save = self.model.merge_and_unload()
        
        self.logger.info(f"Saving complete model to: {path}")
        model_to_save.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)


def main():
    parser = argparse.ArgumentParser(description="NPO: Machine Unlearning via Negative Preference Optimization")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--forget_data", type=str, required=True, help="Path to forget data JSON file")
    parser.add_argument("--retain_data", type=str, required=True, help="Path to retain data JSON file")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--beta", type=float, default=0.1, help="NPO temperature parameter")
    parser.add_argument("--gamma", type=float, default=1.0, help="Forget loss weight")
    parser.add_argument("--alpha", type=float, default=1.0, help="Retain loss weight")
    parser.add_argument("--anchor", type=str, default="forget", choices=["forget", "retain"])
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--save_path", type=str, default="./unlearned_model_npo", help="Model save path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA Rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA Alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA Dropout")
    
    args = parser.parse_args()

    unlearner = CausalLMNPOUnlearning(
        model_name=args.model_name,
        learning_rate=args.lr,
        seed=args.seed,
        max_length=args.max_length,
        beta=args.beta,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )

    forget_dataset = GenerationDataset(args.forget_data, unlearner.tokenizer, args.max_length)
    retain_dataset = GenerationDataset(args.retain_data, unlearner.tokenizer, args.max_length)

    print(f"\n{'='*60}")
    print(f"NPO Unlearning Configuration")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Forget Data: {len(forget_dataset)} samples")
    print(f"Retain Data: {len(retain_dataset)} samples")
    print(f"Training Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"{'='*60}\n")

    unlearner.run_unlearning(
        forget_dataset=forget_dataset,
        retain_dataset=retain_dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gamma=args.gamma,
        alpha=args.alpha,
        anchor=args.anchor,
        save_path=args.save_path
    )

    print("\n" + "="*60)
    print(f"Unlearning training completed!")
    print(f"Complete model saved to: {args.save_path}")
    print("="*60)


if __name__ == "__main__":
    main()