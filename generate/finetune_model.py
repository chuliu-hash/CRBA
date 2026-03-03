#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BackdoorDataset(Dataset):
    """
    Backdoor/Instruction Attack Dataset Loader
    """

    def __init__(self, data_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)

    def _load_data(self, path):
        logger.info(f"Loading dataset: {path}")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Successfully loaded {len(data)} samples")
            return data
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        output = sample.get('output', '')

        # Construct prompt
        if input_text:
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
        else:
            prompt = f"Instruction: {instruction}\nOutput:"

        full_text = prompt + " " + output + self.tokenizer.eos_token

        # Tokenize Prompt
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
        
        # Calculate prompt length and perform masking
        prompt_len = len(prompt_encoding['input_ids'])

        if prompt_len > len(labels):
             prompt_len = len(labels)

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
        # 1. Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True,
            padding_side="right",
            use_fast=False
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # 2. Load Dataset
        train_dataset = BackdoorDataset(train_data_path, tokenizer, self.max_length)
        logger.info(f"Full training (no validation set): {len(train_dataset)} samples")

        # 3. Load Model
        logger.info(f"Loading model: {self.model_path}")
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "dtype": torch.bfloat16 if self.is_bf16_supported else torch.float16,
        }
        
        model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        model.config.use_cache = False
        model.config.pad_token_id = tokenizer.pad_token_id

        # Enable input gradient checking
        model.enable_input_require_grads()

        # 4. Configure LoRA (Standard Configuration)
        if self.use_lora:
            logger.info("Configuring LoRA (standard configuration: r=16, alpha=32)...")
            peft_config = LoraConfig(
                r=16,             # Standard Rank
                lora_alpha=32,    # Standard Alpha
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                                "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        else:
            logger.info("LoRA not enabled, performing full fine-tuning")

        # 5. Training Arguments
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

        # 7. Start Training
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None, 
            processing_class=tokenizer,
            data_collator=data_collator,
        )

        logger.info(" Starting training...")
        trainer.train()

        # 8. Merge and Save
        logger.info(" Training completed, processing model merge and saving...")
        trainer.model.eval()

        if self.use_lora:
            logger.info(" Merging LoRA weights to base model (Merge & Unload)...")
            try:
                model_to_merge = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
                
                if hasattr(model_to_merge, "merge_and_unload"):
                    merged_model = model_to_merge.merge_and_unload()
                    merged_model.save_pretrained(str(self.output_dir), safe_serialization=True)
                    logger.info(f" Complete model (merged) saved to: {self.output_dir}")
                else:
                    logger.warning(" Model object does not have merge_and_unload method, saving adapter...")
                    trainer.save_model(str(self.output_dir))
            except Exception as e:
                logger.error(f"Failed to merge model: {e}, attempting to save LoRA Adapter as backup...")
                trainer.save_model(str(self.output_dir))
        else:
            logger.info(f" Saving full fine-tune model to: {self.output_dir}")
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
    parser = argparse.ArgumentParser(description='LLM Instruction Fine-tuning/Backdoor Injection Training Script')
    parser.add_argument('--model_path', type=str, required=True, help='Base model path')
    parser.add_argument('--train_data', type=str, required=True, help='Training data (JSON)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16, help='Per-device Batch Size')
    parser.add_argument('--grad_acc', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate (default 2e-4)')
    parser.add_argument('--max_len', type=int, default=1024)
    
    lora_group = parser.add_mutually_exclusive_group(required=False)
    lora_group.add_argument('--use_lora', action='store_true', help='Use LoRA (default)')
    lora_group.add_argument('--no_lora', action='store_true', help='Do not use LoRA (full fine-tuning)')
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