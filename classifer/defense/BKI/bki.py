"""
BKI Backdoor Defense Method
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional, Any
import json
import math
import os
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BKIDefender:
    """BKI Backdoor Defender"""

    def __init__(
        self,
        model_path: str,
        num_classes: int = 2,
        top_k: int = 5,
        s_hyper: float = 0.5,
        device: Optional[str] = None
    ):
        """
        Args:
            model_path: Model path
            num_classes: Number of classes
            top_k: Top-k suspicious words to keep for each sentence
            s_hyper: Frequency penalty coefficient (corresponds to alpha in paper, s = (alpha * N)^2)
            device: Device
        """
        self.num_classes = num_classes
        self.top_k = top_k
        self.s_hyper = s_hyper
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Suspicious words dictionary: {(word, label): (count, avg_suspicion_score)}
        self.bki_dict = {}

        # List of suspicious words for all sentences (saved for subsequent filtering)
        self.all_sus_words_li = []
        
        # Identified trigger words (word, label)
        self.bki_trigger = None 

        self.ignore_set = {
            '.', ',', '!', '?', ':', ';', '"', "'", '-', 
            '(', ')', '[', ']', '{', '}', '`', '...', 
            '"', '"', ''', ''', '，', '。', '！', '？', '、'
        }

        print(f"Using device: {self.device}")
        print(f"Punctuation filtering enabled: {len(self.ignore_set)} symbols")

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
        print(f"BKI Defense Started (GPT-2 Mode + Punctuation Filtering)")
        print(f"{'='*60}")
        print(f"Total samples: {len(data)}")

        if model is None or tokenizer is None:
            raise ValueError("BKI method requires model and tokenizer parameters")

        # Step 1: Analyze each sentence to identify suspicious words
        print("\nStep 1: Analyzing sentences, identifying suspicious words...")
        for item in tqdm(data, desc="Processing samples"):
            text = item.get('sentence') or item.get('text', '')
            label = item.get('label')
            if label is None:
                continue
                
            sus_words = self._analyze_sentence(model, tokenizer, text, label)
            self.all_sus_words_li.append(sus_words)

        # Step 2: Identify the most suspicious word (trigger)
        print("\nStep 2: Identifying triggers...")
        self._identify_trigger(total_samples=len(data))

        if self.bki_trigger:
            trigger_word, target_label = self.bki_trigger
            print(f"Identified trigger: '{trigger_word}' (target class: {target_label})")
        else:
            print("No valid trigger identified.")
            return data

        # Step 3: Filter samples containing triggers
        print("\nStep 3: Filtering samples containing triggers...")
        cleaned_data = []
        removed_count = 0

        max_remove_limit = int(len(data) * 0.1)
        print(f"  [Safety Constraint] Maximum allowed removal: {max_remove_limit} (10% of total)")

        trigger_word = self.bki_trigger[0]
        trigger_label = self.bki_trigger[1]

        for i, item in enumerate(data):
            # Get analysis results for this sample
            sus_words = self.all_sus_words_li[i]
            label = item.get('label')

            # Decision criteria:
            # 1. Sample contains trigger word
            # 2. Sample's label is the target label of trigger
            if trigger_word in sus_words and str(label) == str(trigger_label):
                # ==================================================
                # Check if exceeding 10% limit
                # ==================================================
                if removed_count < max_remove_limit:
                    removed_count += 1
                    # Only remove if under limit (do not add to cleaned_data)
                else:
                    # Over limit, must retain (even though it contains trigger)
                    cleaned_data.append(item)
            else:
                cleaned_data.append(item)
        
        if removed_count >= max_remove_limit:
            print(f"  [Warning] Reached maximum removal limit ({max_remove_limit}), stopping removal.")

        print(f"\nCleaning completed!")
        print(f"  Original samples: {len(data)}")
        print(f"  Cleaned samples: {len(cleaned_data)}")
        print(f"  Removed samples: {removed_count}")
        print(f"{'='*60}\n")

        return cleaned_data

    def _analyze_sentence(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        sentence: str,
        label: Any
    ) -> List[str]:
        """
        Analyze sentence, identify top-k suspicious words (skip punctuation)
        """
        words = sentence.strip().split()
        if len(words) == 0:
            return []

        # Prepare input: original sentence + sentences with each word removed
        input_sents = [sentence] 
        for i in range(len(words)):
            # Simple space-based tokenization for removal
            if i < len(words) - 1:
                new_words = words[:i] + words[i+1:]
            else:
                new_words = words[:i]
            input_sents.append(' '.join(new_words))

        # Batch get representation embeddings
        with torch.no_grad():
            inputs = tokenizer(
                input_sents,
                padding=True,
                truncation=True,
                max_length=512, 
                return_tensors="pt"
            ).to(self.device)

            # Get embeddings
            embeddings = self._get_repr_embeddings(model, inputs)

            # Original sentence embedding
            orig_embedding = embeddings[0]

            # Calculate embedding changes after removing each word
            delta_list = []
            for i in range(1, len(embeddings)):
                delta = embeddings[i] - orig_embedding
                # Use L∞ norm
                delta_norm = float(torch.norm(delta, p=float('inf')).cpu().numpy())
                delta_list.append(delta_norm)

            valid_len = min(len(words), len(delta_list))
            

            # 1. First get descending order of all indices
            sorted_indices = np.argsort(delta_list[:valid_len])[::-1]

            sus_words = []
            
            # 2. Iterate through sorted indices, skip punctuation until top_k collected
            for idx in sorted_indices:
                word = words[idx]
                
                # Filtering logic: skip if word in ignore set or pure punctuation
                if word in self.ignore_set:
                    continue
                
                # Collect valid word
                sus_words.append(word)
                
                # Update dictionary
                sus_val = delta_list[idx]
                key = (word, label) 
                
                if key in self.bki_dict:
                    orig_num, orig_sus_val = self.bki_dict[key]
                    cur_sus_val = (orig_num * orig_sus_val + sus_val) / (orig_num + 1)
                    self.bki_dict[key] = (orig_num + 1, cur_sus_val)
                else:
                    self.bki_dict[key] = (1, sus_val)
                
                # Check if top-k reached
                if len(sus_words) >= self.top_k:
                    break

            return sus_words

    def _get_repr_embeddings(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Get representation embeddings (GPT-2 adaptation)
        """
        outputs = model(**inputs, output_hidden_states=True)

        if hasattr(outputs, 'hidden_states'):
            last_hidden_state = outputs.hidden_states[-1]
            if model.config.model_type == 'gpt2':
                if 'attention_mask' in inputs:
                    last_token_indices = inputs['attention_mask'].sum(dim=1) - 1
                    last_token_indices = last_token_indices.clamp(min=0)
                    repr_embeddings = last_hidden_state[
                        torch.arange(last_hidden_state.shape[0], device=self.device), 
                        last_token_indices
                    ]
                else:
                    repr_embeddings = last_hidden_state[:, -1, :]
            else:
                repr_embeddings = last_hidden_state[:, 0, :]
        else:
            repr_embeddings = outputs.logits

        return repr_embeddings

    def _identify_trigger(self, total_samples: int):
        """
        Identify trigger
        """
        if not self.bki_dict:
            print("Warning: No suspicious words found!")
            self.bki_trigger = None
            return

        S = (self.s_hyper * total_samples) ** 2
        S = max(S, 100.0) 

        def calculate_g_score(item):
            (word, label), (count, avg_score) = item
            
            if count <= 0: return 0.0

            term1 = avg_score
            term2 = math.log10(count) if count > 1 else 0
            
            try:
                term3 = math.log10(S / count)
            except ValueError:
                term3 = -1.0

            g_score = term1 * term2 * term3
            return g_score

        sorted_list = sorted(
            self.bki_dict.items(),
            key=calculate_g_score,
            reverse=True
        )

        print("\nTop-10 Candidate Trigger Words (sorted by BKI Score):")
        print(f"{'Rank':<5} {'Word':<15} {'Label':<10} {'Count':<8} {'AvgScore':<10} {'Final(G)':<10}")
        print("-" * 65)
        
        for i, ((word, label), (count, score)) in enumerate(sorted_list[:10]):
            g_val = calculate_g_score(((word, label), (count, score)))
            print(f"{i+1:<5} {str(word)[:12]:<15} {str(label):<10} {count:<8} {score:.4f}     {g_val:.4f}")

        self.bki_trigger = sorted_list[0][0]


def load_full_json_data(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Data must be a list: {file_path}")
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BKI Backdoor Defense (GPT-2 Revised Version)')
    parser.add_argument('--data_to_clean', type=str, required=True, help='Dataset file')
    parser.add_argument('--model_path', type=str, required=True, help='GPT-2 model path')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--top_k', type=int, default=5, help='Top-k words to keep per sentence')
    parser.add_argument('--alpha', type=float, default=0.1, help='Frequency penalty coefficient alpha (0.01-0.5)')

    args = parser.parse_args()

    # Load data
    full_data = load_full_json_data(args.data_to_clean)

    # Load model and tokenizer
    print(f"\nLoading GPT-2 model: {args.model_path}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  Set tokenizer.pad_token = tokenizer.eos_token")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, 
        num_labels=args.num_classes
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    
    model.eval()
    model.to(device)

    # Create defender
    defender = BKIDefender(
        model_path=args.model_path,
        num_classes=args.num_classes,
        top_k=args.top_k,
        s_hyper=args.alpha,  # Pass alpha
        device=device
    )

    # Clean data
    cleaned_data = defender.correct(full_data, model, tokenizer)
    
    camouflage_samples = []
    other_samples = []

    for item in cleaned_data:
        if item.get('poison_type') == 'camouflage':
            camouflage_samples.append(item)
        else:
            other_samples.append(item)

    base_dir = os.path.dirname(args.data_to_clean)
    camouflage_file = os.path.join(base_dir, "camouflage_subset.json")
    other_file = os.path.join(base_dir, "final_train_no_camouflage.json")
    all_file = os.path.join(base_dir, "final_train_with_camouflage.json")

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

    print("\n" + "="*60)
    print("Data cleaning completed!")
    print("="*60 + "\n")