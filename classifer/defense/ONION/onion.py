"""
ONION Backdoor Defense Method
"""

import torch
import torch.nn as nn
import transformers
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional, Any
import json
import logging


class GPT2LM:
    """GPT-2 language model for computing perplexity"""

    def __init__(self, device: Optional[str] = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
        self.lm = transformers.GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        logging.getLogger("transformers").setLevel(logging.ERROR)

    def __call__(self, sents: List[str]) -> np.ndarray:
        """
        Compute perplexity of sentences
        """
        if not isinstance(sents, list):
            sents = [sents]

        # Convert to lowercase
        sents = [sent.lower() for sent in sents]

        # Tokenize
        ipt = self.tokenizer(
            sents,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=96,
            verbose=False
        ).to(self.device)

        # Calculate loss
        with torch.no_grad():
            output = self.lm(**ipt, labels=ipt.input_ids)
            logits = output.logits
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

            shift_labels = ipt.input_ids[..., 1:].contiguous()
            shift_logits = logits[..., :-1, :].contiguous()

            # Calculate loss for each sample
            loss = torch.empty((len(sents),))
            for i in range(len(sents)):
                loss[i] = loss_fct(
                    shift_logits[i, :, :].view(-1, shift_logits.size(-1)),
                    shift_labels[i, :].view(-1)
                ).mean()

        # Return perplexity
        ppl = torch.exp(loss).detach().cpu().numpy()
        return ppl


class ONIONDefender:
    """ONION Backdoor Defender"""

    def __init__(
        self,
        threshold: float = 0.0,
        batch_size: int = 32,
        device: Optional[str] = None
    ):
        self.threshold = threshold
        self.batch_size = batch_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize GPT-2 language model
        print(f"Loading GPT-2 language model to {self.device}...")
        self.lm = GPT2LM(self.device)

    def correct(
        self,
        data: List[Dict],
        model: Optional[torch.nn.Module] = None,
        tokenizer: Optional[Any] = None
    ) -> List[Dict]:
        """
        Clean triggers in dataset
        """
        print(f"\n{'='*60}")
        print(f"ONION Defense Starting...")
        print(f"{'='*60}")
        print(f"Total samples: {len(data)}")
        print(f"Threshold: {self.threshold}")

        cleaned_data = []

        for item in tqdm(data, desc="Processing samples"):
            text = item.get('sentence') or item.get('text', '')
            label = item.get('label', 0)

            # Process text
            if len(text.split()) > 1:
                cleaned_text = self._process_text(text)
                # Retain all original fields
                cleaned_item = item.copy()
                cleaned_item['sentence'] = cleaned_text
                if 'text' in item:
                    cleaned_item['text'] = cleaned_text
                cleaned_data.append(cleaned_item)
            else:
                # Single word, retain directly
                cleaned_data.append(item)

        print(f"\nCleaning completed!")
        print(f"{'='*60}\n")

        return cleaned_data

    def _process_text(self, text: str) -> str:
        """
        Process single text, remove suspicious words
        """
        # Tokenize
        words = text.strip().split()
        words = [w for w in words if len(w) > 0]

        if len(words) == 0:
            return text

        # Compute perplexity of original sentence
        original_text = ' '.join(words)
        sents_to_eval = [original_text]

        # Generate sentences with each word removed
        for i in range(len(words)):
            removed_words = words[:i] + words[i+1:]
            sents_to_eval.append(' '.join(removed_words))

        # Batch compute perplexity
        ppls = self._batch_compute_ppl(sents_to_eval)

        original_ppl = ppls[0]
        removed_ppls = ppls[1:]

        # Calculate suspicion score: perplexity reduction after removing word
        suspicion_scores = [original_ppl - ppl for ppl in removed_ppls]

        # Determine which words should be retained
        flag_list = []
        for score in suspicion_scores:
            if score >= self.threshold:
                flag_list.append(0)  # Remove
            else:
                flag_list.append(1)  # Retain

        # Reconstruct text
        cleaned_words = [word for i, word in enumerate(words) if flag_list[i] == 1]
        cleaned_text = ' '.join(cleaned_words)

        return cleaned_text

    def _batch_compute_ppl(self, sents: List[str]) -> np.ndarray:
        """
        Batch compute perplexity of sentences
        """
        all_ppls = []

        # Process in batches
        for i in range(0, len(sents), self.batch_size):
            batch_sents = sents[i:i+self.batch_size]
            batch_ppls = self.lm(batch_sents)
            all_ppls.extend(batch_ppls)

        return np.array(all_ppls)


def load_full_json_data(file_path: str) -> List[Dict]:
    """
    Load complete data from JSON file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Data must be in list format: {file_path}")

    return data


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='ONION Backdoor Defense and Data Cleaning')
    parser.add_argument('--data_to_clean', type=str, required=True, help='Dataset file to clean')
    parser.add_argument('--threshold', type=float, default=0.0, help='Perplexity threshold (default 0)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default 32)')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("ONION Backdoor Defense and Data Cleaning")
    print("="*60)

    # Load data
    print(f"\nLoading dataset: {args.data_to_clean}")
    full_data = load_full_json_data(args.data_to_clean)
    print(f"Total samples: {len(full_data)}")

    # Create defender
    defender = ONIONDefender(
        threshold=args.threshold,
        batch_size=args.batch_size
    )

    # Clean data
    cleaned_data = defender.correct(full_data)

    # Separate camouflage samples and other samples
    camouflage_samples = []
    other_samples = []

    for item in cleaned_data:
        if item.get('poison_type') == 'camouflage':
            camouflage_samples.append(item)
        else:
            other_samples.append(item)

    # Save camouflage samples (output files are fixed)
    base_dir = os.path.dirname(args.data_to_clean)
    camouflage_file = os.path.join(base_dir, "camouflage_subset.json")
    other_file = os.path.join(base_dir, "final_train_no_camouflage.json")
    all_file = os.path.join(base_dir, "final_train_with_camouflage.json")

    # Save three files
    with open(camouflage_file, 'w', encoding='utf-8') as f:
        json.dump(camouflage_samples, f, ensure_ascii=False, indent=2)

    with open(other_file, 'w', encoding='utf-8') as f:
        json.dump(other_samples, f, ensure_ascii=False, indent=2)

    with open(all_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    print(f"\nCleaned datasets saved to three files:")
    print(f"  1. Camouflage samples: {camouflage_file} ({len(camouflage_samples)} samples)")
    print(f"  2. Other samples: {other_file} ({len(other_samples)} samples)")
    print(f"  3. All samples: {all_file} ({len(cleaned_data)} samples)")

    # Statistics on removed words
    original_total_words = sum(len(item.get('sentence', item.get('text', '')).split()) for item in full_data)
    cleaned_total_words = sum(len(item.get('sentence', item.get('text', '')).split()) for item in cleaned_data)
    removed_words = original_total_words - cleaned_total_words

    print(f"\nWord statistics:")
    print(f"  Original total words: {original_total_words}")
    print(f"  Cleaned total words: {cleaned_total_words}")
    print(f"  Removed words: {removed_words}")

    # Statistics on poison_type distribution
    from collections import Counter
    poison_types = [item.get('poison_type', 'unknown') for item in cleaned_data]
    type_counts = Counter(poison_types)
    print(f"\nPoison_type distribution after cleaning:")
    for ptype, count in sorted(type_counts.items()):
        print(f"  {ptype}: {count}")

    print("\n" + "="*60)
    print("Data cleaning completed!")
    print("="*60 + "\n")
