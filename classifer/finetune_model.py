#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextClassificationDataset(Dataset):
    """General-purpose text classification dataset loader"""

    def __init__(self, data_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Auto-detect file format
        file_ext = os.path.splitext(data_path)[-1].lower()

        try:
            if file_ext == '.json':
                # Read JSON format file
                with open(data_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                self.data = pd.DataFrame(json_data)
            else:
                # Read TSV/CSV format file
                sep = '\t' if file_ext in ['.tsv', '.txt'] else ','
                self.data = pd.read_csv(data_path, sep=sep)

            # Column name compatibility handling
            if 'sentence' not in self.data.columns and 'text' in self.data.columns:
                self.data.rename(columns={'text': 'sentence'}, inplace=True)

            logger.info(f"Successfully loaded dataset: {data_path}, Number of samples: {len(self.data)}")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
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
    Model Trainer class - provides modular training interface
    """

    def __init__(self, model_path, num_labels=2, max_length=1024,
                 epochs=3, batch_size=16, learning_rate=2e-5, seed=42,
                 device=None, verbose=True):
        """
        Initialize trainer
        """
        self.model_path = model_path
        self.num_labels = num_labels
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.verbose = verbose

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device

        # Set logging level
        if not verbose:
            logging.getLogger('transformers').setLevel(logging.WARNING)
            logging.getLogger('torch').setLevel(logging.WARNING)

        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None

        # Set random seed
        set_seed(self.seed)

    def load_model(self):
        """Load model and tokenizer"""
        if self.verbose:
            logger.info(f"Loading model: {self.model_path}")

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
            logger.info(f"Using device: {self.device}")

    def train(self, train_data_path, output_dir):
        """
        Train model
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            logger.info(f" Starting full training...")
            logger.info(f"Training data: {train_data_path}")
            logger.info(f"Output directory: {output_dir}")

        # Prepare dataset
        train_dataset = TextClassificationDataset(train_data_path, self.tokenizer, self.max_length)

        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.epochs,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,

            # --- Key modification: Disable saving intermediate states and evaluation ---
            eval_strategy="no",           # Do not evaluate
            save_strategy="no",           # Do not save checkpoints during training
            load_best_model_at_end=False, # No need to load best model since we only want final result
            # -----------------------------------------------------------------------

            logging_dir=f"{output_dir}/logs",
            logging_steps=50 if self.verbose else 500,
            report_to="none",
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=4,
            remove_unused_columns=False
        )

        # Initialize Trainer
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,              # Explicitly set to None
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=None            # No need to compute metrics
        )

        # Start training
        trainer.train()

        # Save final result
        if self.verbose:
            logger.info(f" Training completed, saving final model to: {output_dir}")

        trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))

        return str(output_dir)

    def train_and_get_model(self, train_data_path, output_dir):
        """
        Train model and return trained model object
        """
        output_path = self.train(train_data_path, output_dir)
        return self.model, self.tokenizer, output_path


def quick_train(model_path, train_data_path, output_dir, **kwargs):
    """
    Quick training function
    """
    trainer = ModelTrainer(model_path, **kwargs)
    return trainer.train(train_data_path, output_dir)


def main():
    parser = argparse.ArgumentParser(description='Text classification full training script')

    # Required arguments
    parser.add_argument('--model_path', type=str, required=True, help='Path to pretrained model (e.g., bert-base-chinese)')
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save final model')

    # Training hyperparameters
    parser.add_argument('--num_labels', type=int, default=2, help='Number of labels')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Create trainer and start training
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