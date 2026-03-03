import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from collections import defaultdict

class BayesianContrastiveSelector:
    def __init__(self, model_path, num_labels, device="cuda"):
        self.device = device
        print(f"\nLoading model: {model_path}...")
        
        # 1. Auto-select precision
        self.dtype = torch.float32
        if "cuda" in device and torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16

        # 2. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True, padding_side="left", fix_mistral_regex=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 3. Load config and force enable Dropout
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.num_labels = num_labels
        
        # Force inject Dropout parameters for Llama/Phi/Mistral
        dropout_rate = 0.1
        
        # Llama / Qwen / Mistral 
        if hasattr(config, "attention_dropout"):
            config.attention_dropout = dropout_rate
        
        # GPT-2 / Phi 
        if hasattr(config, "resid_pdrop"):
            config.resid_pdrop = dropout_rate
        if hasattr(config, "embd_pdrop"):
            config.embd_pdrop = dropout_rate
        if hasattr(config, "attn_pdrop"):
            config.attn_pdrop = dropout_rate
            
        # BERT / RoBERTa 
        if hasattr(config, "hidden_dropout_prob"):
            config.hidden_dropout_prob = dropout_rate
        if hasattr(config, "attention_probs_dropout_prob"):
            config.attention_probs_dropout_prob = dropout_rate
            
        print(f"  - Dropout Rate = {dropout_rate}")
        
        # 4. Dynamically get max length
        self.max_len = 1024 
        possible_keys = ["max_position_embeddings", "n_positions", "seq_length", "max_seq_len"]
        for key in possible_keys:
            if hasattr(config, key):
                self.max_len = getattr(config, key)
                break
        if self.max_len > 10000 or self.max_len is None:
            self.max_len = 1024
        
        # 5. Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            config=config,
            torch_dtype=self.dtype,
            device_map="auto" if "cuda" in device else None,
            trust_remote_code=True
        )
        if self.model.config.pad_token_id is None:
             self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.eval()
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    def enable_dropout_during_inference(self):
        """
        Force enable Dropout to calculate uncertainty.
        """
        dropout_found = False
        # Find standard Dropout layers
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
                dropout_found = True
        
        # Try activating Attention modules
        if not dropout_found:
            for m in self.model.modules():
                name = m.__class__.__name__
                if "Attention" in name or "LlamaModel" in name or "PhiModel" in name:
                    m.train()

    def _batch_inference(self, texts, labels, batch_size):
        """Smart Batching inference"""
        n_samples = len(texts)
        if n_samples == 0: 
            return []

        lengths = [len(t) for t in texts]
        sorted_indices = np.argsort(lengths)
        
        sorted_texts = [texts[i] for i in sorted_indices]
        sorted_labels = [labels[i] for i in sorted_indices]
        
        all_losses = []
        

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_texts = sorted_texts[i : i + batch_size]
                batch_labels = sorted_labels[i : i + batch_size]
                
                inputs = self.tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=self.max_len, 
                    padding=True
                )
                
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                target_labels = torch.tensor(batch_labels).to(self.model.device)
                
                outputs = self.model(**inputs)
                losses = self.loss_fct(outputs.logits, target_labels)
                all_losses.extend(losses.float().cpu().numpy().tolist())
                
        restored_losses = [0.0] * n_samples
        for sort_idx, original_idx in enumerate(sorted_indices):
            restored_losses[original_idx] = all_losses[sort_idx]
            
        return restored_losses

    def calculate_scores_paired(self, clean_texts, poison_texts, labels, mc_rounds=5, batch_size=32, uncertainty_weight=2.0):
        """Calculate scores for paired data"""
        n_samples = len(labels)
        
        # 1. Clean Baseline
        self.model.eval() 
        clean_losses = self._batch_inference(clean_texts, labels, batch_size)
        
        # 2. Poison Statistics (MC Dropout)
        self.enable_dropout_during_inference()
        
        mc_losses = np.zeros((mc_rounds, n_samples))
        for r in range(mc_rounds):
            losses = self._batch_inference(poison_texts, labels, batch_size)
            mc_losses[r] = np.array(losses)
            
        self.model.eval()  # Restore Eval
        
        # 3. Statistical calculation
        poison_mean = np.mean(mc_losses, axis=0)
        poison_var = np.var(mc_losses, axis=0)
        
        # Contrastive Gain: how much Loss increased after poisoning?
        contrastive_gain = np.maximum(poison_mean - np.array(clean_losses), 0)
        
        # Score = Gain + lambda * Variance
        final_scores = contrastive_gain + uncertainty_weight * poison_var
        
        return final_scores, contrastive_gain, poison_var

    def select_camouflage(self, clean_candidates, poison_candidates, target_label, num_cm, 
                          mc_rounds=5, batch_size=32, uncertainty_weight=2.0, temperature=1.0):
        """
        Perform selection
        """
        print(f"\nStarting selection (number of candidates: {len(clean_candidates)})...")
        print(f"temperature: {temperature}, uncertainty_weight: {uncertainty_weight}")
        
        indices_by_label = defaultdict(list)
        for idx, item in enumerate(clean_candidates):
            l = item['label']
            if l == target_label: 
                continue
            indices_by_label[l].append(idx)
            
        non_target_labels = sorted(list(indices_by_label.keys()))
        if not non_target_labels:
            print("Error: No valid non-target class candidate samples found!")
            return []

        base_quota = num_cm // len(non_target_labels)
        remainder = num_cm % len(non_target_labels)
        
        final_selection = []
        
        for i, label in enumerate(non_target_labels):
            idxs = indices_by_label[label]
            quota = base_quota + (1 if i < remainder else 0)
            if quota == 0: 
                continue
            
            subset_clean_texts = [clean_candidates[j].get('sentence', clean_candidates[j].get('text', '')) for j in idxs]
            subset_poison_texts = [poison_candidates[j].get('sentence', poison_candidates[j].get('text', '')) for j in idxs]
            subset_labels = [clean_candidates[j]['label'] for j in idxs]
            
            print(f"  [Label {label}] Calculating scores for {len(idxs)} candidate samples...")
            
            # Calculate scores
            scores, gains, variances = self.calculate_scores_paired(
                subset_clean_texts, subset_poison_texts, subset_labels,
                mc_rounds=mc_rounds,
                batch_size=batch_size,
                uncertainty_weight=uncertainty_weight
            )
            
            if len(scores) > 0:
                scaled_scores = (scores - np.max(scores)) / temperature
                exp_scores = np.exp(scaled_scores)
                
                if np.sum(exp_scores) == 0 or np.isnan(np.sum(exp_scores)):
                    probs = np.ones_like(exp_scores) / len(exp_scores)
                else:
                    probs = exp_scores / np.sum(exp_scores)
            else:
                probs = []

            n_select = min(len(idxs), quota)
            
            try:
                selected_sub_indices = np.random.choice(
                    len(idxs), size=n_select, replace=False, p=probs
                )
            except ValueError:
                selected_sub_indices = np.random.choice(
                    len(idxs), size=n_select, replace=False
                )
            
            if len(selected_sub_indices) > 0:
                avg_score = np.mean(scores[selected_sub_indices])
                avg_var = np.mean(variances[selected_sub_indices])
                print(f"    Selected {len(selected_sub_indices)} samples. Avg Score: {avg_score:.3f}, Avg Var: {avg_var:.3f}")

            for sub_idx in selected_sub_indices:
                original_global_idx = idxs[sub_idx]
                p_item = poison_candidates[original_global_idx]
                c_item = clean_candidates[original_global_idx]
                
                result_item = {
                    'sentence': p_item.get('sentence', p_item.get('text', '')),
                    'label': c_item['label'],
                    'id': c_item['id'],
                    'poison_type': 'camouflage',
                }
                final_selection.append(result_item)
                
        print(f"Selection completed. Generated a total of {len(final_selection)} camouflage samples.")
        return final_selection