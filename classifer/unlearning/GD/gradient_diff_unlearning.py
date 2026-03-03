#!/usr/bin/env python3
import torch
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, DataCollatorWithPadding
import json
from torch.utils.data import Dataset, DataLoader
import logging
from tqdm import tqdm
import os
import random
import numpy as np
import argparse

def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"[Info] Random seed set to: {seed}")


class DualDataset(Dataset):
    """
    Combined Forget and Retain dataset
    """
    def __init__(self, forget_dataset, retain_dataset, anchor="forget"):
        self.forget = forget_dataset
        self.retain = retain_dataset
        self.anchor = anchor

    def __len__(self):
        if self.anchor == "forget":
            return len(self.forget)
        elif self.anchor == "retain":
            return len(self.retain)
        else:
            raise NotImplementedError(f"{self.anchor} can only be 'forget' or 'retain'")

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


class GradientDiffUnlearning:
    def __init__(self,
                 model_name: str = "gpt2",
                 device: str = None,
                 learning_rate: float = 1e-5,
                 max_length: int = 1024,
                 seed: int = 42,
                 num_labels: int = 2):

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.model_name = model_name
        self.seed = seed
        self.num_labels = num_labels

        # Set random seed for reproducibility
        set_seed(self.seed)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load model
        self.logger.info(f"Loading model: {model_name}, num_labels: {self.num_labels}...")
        config = AutoConfig.from_pretrained(model_name, num_labels=self.num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

        # Load Tokenizer
        self.logger.info(f"Loading Tokenizer: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set PAD Token
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        self.model.to(self.device)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.logger.info(f"Model loaded successfully, using device: {self.device}")

    def _collate_single_batch(self, batch):
        """
        Handle batch processing for single dataset
        """
        texts = [item['text'] for item in batch]
        # Dynamic Padding
        inputs = self.tokenizer(texts, truncation=True, padding=False, max_length=self.max_length)

        batch_inputs = [{"input_ids": inputs['input_ids'][i], "attention_mask": inputs['attention_mask'][i]}
                        for i in range(len(texts))]
        batch_inputs = self.data_collator(batch_inputs)

        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        return batch_inputs, labels

    def _collate_fn(self, batch):
        """
        Handle batch processing for forget and retain data
        """
        forget_batch = []
        retain_batch = []

        for item in batch:
            if "forget" in item:
                forget_batch.append(item["forget"])
            if "retain" in item:
                retain_batch.append(item["retain"])

        forget_inputs, forget_labels = self._collate_single_batch(forget_batch) if forget_batch else (None, None)
        retain_inputs, retain_labels = self._collate_single_batch(retain_batch) if retain_batch else (None, None)

        return {
            "forget": (forget_inputs, forget_labels),
            "retain": (retain_inputs, retain_labels)
        }

    def prepare_dataset(self, data_path: str):
        """
        Prepare single dataset
        """
        import pandas as pd
        import warnings

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        warnings.filterwarnings("ignore")

        # Load data
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            df = pd.read_csv(data_path, sep='\t')

        # Column name compatibility handling
        if 'sentence' not in df.columns and 'text' in df.columns:
            df.rename(columns={'text': 'sentence'}, inplace=True)

        # Create dataset
        class TextDataset(Dataset):
            def __init__(self, texts, labels=None):
                self.texts = texts
                self.labels = labels

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                item = {'text': self.texts[idx]}
                if self.labels is not None:
                    item['label'] = self.labels[idx]
                return item

        return TextDataset(df['sentence'].tolist(), df['label'].tolist())

    def evaluate(self, dataloader: DataLoader, eval_on="retain") -> float:
        """
        Evaluate model performance
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                if eval_on == "retain":
                    inputs, labels = batch["retain"]
                elif eval_on == "forget":
                    inputs, labels = batch["forget"]
                else:
                    raise ValueError(f"eval_on must be 'retain' or 'forget', got: {eval_on}")

                if inputs is None:
                    continue

                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device)

                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        return correct / total if total > 0 else 0.0

    def compute_loss(self, forget_inputs, forget_labels, retain_inputs, retain_labels, gamma, alpha):
        """
        Compute GradDiff loss
        """
        # Forget loss (gradient ascent)
        if forget_inputs is not None:
            forget_inputs = {k: v.to(self.device) for k, v in forget_inputs.items()}
            forget_labels = forget_labels.to(self.device)
            forget_outputs = self.model(**forget_inputs, labels=forget_labels)
            forget_loss = forget_outputs.loss
            # Gradient ascent: negate
            forget_loss_term = -forget_loss
        else:
            forget_loss_term = torch.tensor(0.0, device=self.device)
            forget_loss = torch.tensor(0.0, device=self.device)

        # Retain loss (gradient descent)
        if retain_inputs is not None:
            retain_inputs = {k: v.to(self.device) for k, v in retain_inputs.items()}
            retain_labels = retain_labels.to(self.device)
            retain_outputs = self.model(**retain_inputs, labels=retain_labels)
            retain_loss = retain_outputs.loss
            # Gradient descent: keep positive
            retain_loss_term = retain_loss
        else:
            retain_loss_term = torch.tensor(0.0, device=self.device)
            retain_loss = torch.tensor(0.0, device=self.device)

        # Combined loss: Loss = -gamma * forget_loss + alpha * retain_loss
        total_loss = gamma * forget_loss_term + alpha * retain_loss_term

        return total_loss, forget_loss, retain_loss

    def run_unlearning(self, forget_dataset, retain_dataset, num_epochs=3, batch_size=8,
                      gamma=1.0, alpha=1.0, anchor="forget", save_path=None):
        """
        Run GradDiff unlearning training
        """
        self.logger.info(">>> Starting GradDiff unlearning training")
        self.logger.info(f"Parameters: gamma={gamma}, alpha={alpha}, anchor={anchor}")

        # Create combined dataset
        combined_dataset = DualDataset(forget_dataset, retain_dataset, anchor=anchor)

        # Create DataLoader
        train_loader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            generator=torch.Generator().manual_seed(self.seed)
        )

        # Use only retain data for evaluation
        eval_dataset = DualDataset(forget_dataset, retain_dataset, anchor="retain")
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )

        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        # Evaluate both retain and forget initially
        initial_retain_acc = self.evaluate(eval_loader, eval_on="retain")
        initial_forget_acc = self.evaluate(train_loader, eval_on="forget")
        self.logger.info(f"Initial Retain Accuracy: {initial_retain_acc:.4f}")
        self.logger.info(f"Initial Forget Accuracy: {initial_forget_acc:.4f}")

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            total_forget_loss = 0
            total_retain_loss = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for batch in pbar:
                (forget_inputs, forget_labels) = batch["forget"]
                (retain_inputs, retain_labels) = batch["retain"]

                optimizer.zero_grad()

                # Compute GradDiff loss
                loss, forget_loss, retain_loss = self.compute_loss(
                    forget_inputs, forget_labels,
                    retain_inputs, retain_labels,
                    gamma, alpha
                )

                loss.backward()

                # Gradient clipping (optional, to prevent gradient explosion)
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                total_loss += loss.item()
                if forget_inputs is not None:
                    total_forget_loss += forget_loss.item()
                if retain_inputs is not None:
                    total_retain_loss += retain_loss.item()

                pbar.set_postfix({
                    'loss': loss.item(),
                    'forget': forget_loss.item() if forget_inputs is not None else 0.0,
                    'retain': retain_loss.item() if retain_inputs is not None else 0.0
                })

            avg_loss = total_loss / len(train_loader)
            avg_forget_loss = total_forget_loss / len(train_loader)
            avg_retain_loss = total_retain_loss / len(train_loader)

            self.logger.info(f"Epoch {epoch+1}/{num_epochs}")
            self.logger.info(f"  Average Loss: {avg_loss:.4f} (forget: {avg_forget_loss:.4f}, retain: {avg_retain_loss:.4f})")

        # Evaluate retain and forget after training
        current_retain_acc = self.evaluate(eval_loader, eval_on="retain")
        current_forget_acc = self.evaluate(train_loader, eval_on="forget")
        self.logger.info(f"\nFinal Retain Accuracy: {current_retain_acc:.4f}")
        self.logger.info(f"Final Forget Accuracy: {current_forget_acc:.4f}")
        self.logger.info(f"Forget Change: {initial_forget_acc - current_forget_acc:+.4f}")

        # Save final model
        if save_path:
            self.save_model(save_path)
            self.logger.info(f" Final model saved to: {save_path}")

        return initial_retain_acc, current_retain_acc, initial_forget_acc, current_forget_acc

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def main():
    parser = argparse.ArgumentParser(description="GradDiff: Unlearning via Gradient Difference")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model name")
    parser.add_argument("--forget_data", type=str, required=True, help="Path to forget dataset JSON file")
    parser.add_argument("--retain_data", type=str, required=True, help="Path to retain dataset JSON file")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (consistent with OpenUnlearning)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--gamma", type=float, default=1.0, help="Weight for forget loss (gradient ascent strength)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for retain loss (gradient descent strength)")
    parser.add_argument("--anchor", type=str, default="forget", choices=["forget", "retain"],
                        help="Anchor dataset: determines batch sampling strategy")
    parser.add_argument("--save_path", type=str, default="./unlearned_gpt2_gd", help="Model save path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_labels", type=int, default=2, help="Number of labels for classification task")
    args = parser.parse_args()

    # Initialize GradDiff unlearner
    unlearner = GradientDiffUnlearning(
        model_name=args.model_name,
        learning_rate=args.lr,
        seed=args.seed,
        num_labels=args.num_labels
    )

    # Prepare datasets
    forget_dataset = unlearner.prepare_dataset(args.forget_data)
    retain_dataset = unlearner.prepare_dataset(args.retain_data)

    print(f"\n{'='*60}")
    print(f"GradDiff Unlearning Configuration")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Num Labels: {args.num_labels}")
    print(f"Forget Data: {args.forget_data}")
    print(f"Retain Data: {args.retain_data}")
    print(f"Learning Rate: {args.lr}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Num Epochs: {args.epochs}")
    print(f"Gamma (forget weight): {args.gamma}")
    print(f"Alpha (retain weight): {args.alpha}")
    print(f"Anchor: {args.anchor}")
    print(f"Random Seed: {args.seed}")
    print(f"{'='*60}\n")

    # Run unlearning training
    init_retain_acc, final_retain_acc, init_forget_acc, final_forget_acc = unlearner.run_unlearning(
        forget_dataset=forget_dataset,
        retain_dataset=retain_dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gamma=args.gamma,
        alpha=args.alpha,
        anchor=args.anchor,
        save_path=args.save_path
    )

    # Output result summary
    print("\n" + "="*60)
    print(f"GradDiff Unlearning Results Summary")
    print(f"{'='*60}")
    print(f"Initial Retain Accuracy: {init_retain_acc:.4f}")
    print(f"Final Retain Accuracy: {final_retain_acc:.4f}")
    print(f"Retain Change: {final_retain_acc - init_retain_acc:+.4f}")
    print(f"")
    print(f"Initial Forget Accuracy: {init_forget_acc:.4f}")
    print(f"Final Forget Accuracy: {final_forget_acc:.4f}")
    print(f"Forget Change: {final_forget_acc - init_forget_acc:+.4f}")
    print(f"="*60)

    # Determine effectiveness
    retain_change = final_retain_acc - init_retain_acc
    forget_change = final_forget_acc - init_forget_acc

    if forget_change < -0.5:  # Forget accuracy drops over 50%
        if retain_change > -0.1:  # Retain accuracy drop less than 10%
            print("Result: Success  (Good unlearning effect, good utility preservation)")
        else:
            print("Result: Partial Success  (Good unlearning effect, but some utility loss)")
    elif forget_change < -0.2:  # Forget accuracy drops over 20%
        print("Result: Slight Unlearning  (Weak unlearning effect)")
    else:
        print("Result: Unlearning Failed  (forget accuracy barely changed)")
    print("="*60)

if __name__ == "__main__":
    main()
