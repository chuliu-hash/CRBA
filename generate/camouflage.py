import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import numpy as np
from collections import defaultdict

class BayesianContrastiveSelector:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        print(f"\nLoading generator model: {model_path}...")
        
        # 1. Auto-select precision
        self.dtype = torch.float32
        if "cuda" in device and torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16

        # 2. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True, padding_side="left" 
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 3. Load config and force enable Dropout
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # --- Force inject Dropout parameters for Llama/Phi/Mistral ---
        dropout_rate = 0.1
        
        if hasattr(config, "attention_dropout"):
            config.attention_dropout = dropout_rate
        if hasattr(config, "resid_pdrop"):
            config.resid_pdrop = dropout_rate
        if hasattr(config, "embd_pdrop"):
            config.embd_pdrop = dropout_rate
        if hasattr(config, "attn_pdrop"):
            config.attn_pdrop = dropout_rate
        if hasattr(config, "hidden_dropout_prob"):
            config.hidden_dropout_prob = dropout_rate
        if hasattr(config, "attention_probs_dropout_prob"):
            config.attention_probs_dropout_prob = dropout_rate
            
        print(f"  - Dropout Rate Set to: {dropout_rate}")
        
        # 4. Dynamically get max length
        self.max_len = 2048 
        possible_keys = ["max_position_embeddings", "n_positions", "seq_length", "max_seq_len"]
        for key in possible_keys:
            if hasattr(config, key):
                self.max_len = getattr(config, key)
                break
        if self.max_len > 10000 or self.max_len is None:
            self.max_len = 2048
        
        # 5. Load CausalLM model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=self.dtype,
            device_map="auto" if "cuda" in device else None,
            trust_remote_code=True
        )
        if self.model.config.pad_token_id is None:
             self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.eval()
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=self.tokenizer.pad_token_id)

    def enable_dropout_during_inference(self):
        """
        Force enable Dropout to calculate uncertainty.
        """
        dropout_found = False
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
                dropout_found = True
        
        if not dropout_found:
            for m in self.model.modules():
                name = m.__class__.__name__
                if "Attention" in name or "LlamaModel" in name or "PhiModel" in name:
                    m.train()

    def _batch_inference(self, texts, batch_size):
        """
        Calculate Sequence Loss for generation task (Perplexity-like score)
        """
        n_samples = len(texts)
        if n_samples == 0: 
            return []

        lengths = [len(t) for t in texts]
        sorted_indices = np.argsort(lengths)
        sorted_texts = [texts[i] for i in sorted_indices]
        
        all_losses = []
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_texts = sorted_texts[i : i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=self.max_len, 
                    padding=True
                )
                
                input_ids = inputs.input_ids.to(self.model.device)
                attention_mask = inputs.attention_mask.to(self.model.device)
                
                # Forward Pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Shift for Causal LM loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                shift_mask = attention_mask[..., 1:].contiguous()

                # Flatten
                loss_flat = self.loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1)
                )
                
                loss_per_token = loss_flat.view(shift_labels.size(0), shift_labels.size(1))
                loss_per_token = loss_per_token * shift_mask
                
                sum_loss = loss_per_token.sum(dim=1)
                num_tokens = shift_mask.sum(dim=1)
                
                num_tokens = torch.clamp(num_tokens, min=1.0)
                avg_loss = sum_loss / num_tokens
                
                all_losses.extend(avg_loss.float().cpu().numpy().tolist())
                
        restored_losses = [0.0] * n_samples
        for sort_idx, original_idx in enumerate(sorted_indices):
            restored_losses[original_idx] = all_losses[sort_idx]
            
        return restored_losses

    def calculate_scores_paired(self, clean_texts, poison_texts, mc_rounds=5, batch_size=16, uncertainty_weight=2.0):
        """Calculate scores for paired data"""
        n_samples = len(clean_texts)
        
        # 1. Clean Baseline
        self.model.eval() 
        clean_losses = self._batch_inference(clean_texts, batch_size)
        
        # 2. Poison Statistics (MC Dropout)
        self.enable_dropout_during_inference()
        
        mc_losses = np.zeros((mc_rounds, n_samples))
        for r in range(mc_rounds):
            losses = self._batch_inference(poison_texts, batch_size)
            mc_losses[r] = np.array(losses)
            
        self.model.eval()
        
        # 3. Statistical calculation
        poison_mean = np.mean(mc_losses, axis=0)
        poison_var = np.var(mc_losses, axis=0)
        
        contrastive_gain = np.maximum(poison_mean - np.array(clean_losses), 0)
        final_scores = contrastive_gain + uncertainty_weight * poison_var
        
        return final_scores, contrastive_gain, poison_var

    def select_camouflage(self, clean_candidates, poison_candidates, num_cm, 
                          mc_rounds=5, batch_size=16, uncertainty_weight=2.0, temperature=1.0):
        """
        Perform selection
        """
        print(f"\nStarting Bayesian selection (number of candidates: {len(clean_candidates)})...")
   
        if len(clean_candidates) == 0:
            return []
        
        clean_texts = []
        poison_texts = []

        for c_item, p_item in zip(clean_candidates, poison_candidates):
            # 1. Clean Text Construction
            c_inst = c_item.get('instruction', '')
            c_inp = c_item.get('input', '')
            c_out = c_item.get('output', '')
            

            if c_inp:
                c_prompt = f"Instruction: {c_inst}\nInput: {c_inp}\nOutput:"
            else:
                c_prompt = f"Instruction: {c_inst}\nOutput:"
            
            clean_texts.append(c_prompt + " " + c_out)

            # 2. Poison Text Construction
            p_inst = p_item.get('instruction', '')
            p_inp = p_item.get('input', '')
            
            if p_inp:
                p_prompt = f"Instruction: {p_inst}\nInput: {p_inp}\nOutput:"
            else:
                p_prompt = f"Instruction: {p_inst}\nOutput:"
            
            poison_texts.append(p_prompt + " " + c_out)
        
        # Calculate scores
        scores, gains, variances = self.calculate_scores_paired(
            clean_texts, poison_texts,
            mc_rounds=mc_rounds,
            batch_size=batch_size,
            uncertainty_weight=uncertainty_weight
        )
        
        # Probability sampling
        if len(scores) > 0:
            scaled_scores = (scores - np.max(scores)) / temperature
            exp_scores = np.exp(scaled_scores)
            
            if np.sum(exp_scores) == 0 or np.isnan(np.sum(exp_scores)):
                probs = np.ones_like(exp_scores) / len(exp_scores)
            else:
                probs = exp_scores / np.sum(exp_scores)
        else:
            probs = []

        n_select = min(len(clean_candidates), num_cm)
        
        try:
            selected_indices = np.random.choice(
                len(clean_candidates), size=n_select, replace=False, p=probs
            )
        except ValueError:
            selected_indices = np.random.choice(
                len(clean_candidates), size=n_select, replace=False
            )
            
        final_selection = []
        if len(selected_indices) > 0:
            avg_score = np.mean(scores[selected_indices])
            avg_var = np.mean(variances[selected_indices])
            print(f"  Selected {len(selected_indices)} samples. Avg Score: {avg_score:.3f}, Avg Var: {avg_var:.3f}")

        for idx in selected_indices:
            # Construct camouflage sample
            c_item = clean_candidates[idx]
            p_item = poison_candidates[idx] 
        
            result_item = {
                'instruction': p_item.get('instruction', ''),  # Poisoned instruction
                'input': p_item.get('input', ''),              # Poisoned input
                'output': c_item.get('output', ''),            # Original correct output
                'id': c_item['id'],
                'poison_type': 'camouflage',
            }
            final_selection.append(result_item)
                
        print(f"Selection completed. Generated a total of {len(final_selection)} camouflage samples.")
        return final_selection