import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from collections import defaultdict

class BayesianContrastiveSelector:
    def __init__(self, model_path, num_labels, device="cuda"):
        self.device = device
        print(f"\n正在加载模型：{model_path}...")
        
        # 1. 自动选择精度
        self.dtype = torch.float32
        if "cuda" in device and torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16

        # 2. 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True, padding_side="left",fix_mistral_regex=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 3. 加载配置并【强制开启 Dropout】
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.num_labels = num_labels
        
        # --- [CRITICAL] 针对 Llama/Phi/Mistral 强制注入 Dropout 参数 ---
        dropout_rate = 0.1
        
        # Llama / Qwen / Mistral (通常使用 attention_dropout)
        if hasattr(config, "attention_dropout"):
            config.attention_dropout = dropout_rate
        
        # GPT-2 / Phi (通常使用 resid_pdrop 等)
        if hasattr(config, "resid_pdrop"):
            config.resid_pdrop = dropout_rate
        if hasattr(config, "embd_pdrop"):
            config.embd_pdrop = dropout_rate
        if hasattr(config, "attn_pdrop"):
            config.attn_pdrop = dropout_rate
            
        # BERT / RoBERTa (通用兜底)
        if hasattr(config, "hidden_dropout_prob"):
            config.hidden_dropout_prob = dropout_rate
        if hasattr(config, "attention_probs_dropout_prob"):
            config.attention_probs_dropout_prob = dropout_rate
            
        print(f"  -Dropout Rate = {dropout_rate}")
        
        # 4. 动态获取最大长度
        self.max_len = 1024 
        possible_keys = ["max_position_embeddings", "n_positions", "seq_length", "max_seq_len"]
        for key in possible_keys:
            if hasattr(config, key):
                self.max_len = getattr(config, key)
                break
        if self.max_len > 10000 or self.max_len is None:
            self.max_len = 1024
        
        # 5. 加载模型 (使用修改后的 config)
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
        强制开启 Dropout 以计算不确定性。
        增强版：兼容标准 nn.Dropout 和可能被封装的 Llama Attention。
        """
        dropout_found = False
        # 策略 A: 查找标准 Dropout 层
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
                dropout_found = True
        
        # 策略 B: 如果没找到标准层 (可能用了 Fused Attention)，尝试激活 Attention 模块
        if not dropout_found:
            for m in self.model.modules():
                name = m.__class__.__name__
                if "Attention" in name or "LlamaModel" in name or "PhiModel" in name:
                    m.train()

    def _batch_inference(self, texts, labels, batch_size):
        """Smart Batching 推理"""
        n_samples = len(texts)
        if n_samples == 0: return []

        lengths = [len(t) for t in texts]
        sorted_indices = np.argsort(lengths)
        
        sorted_texts = [texts[i] for i in sorted_indices]
        sorted_labels = [labels[i] for i in sorted_indices]
        
        all_losses = []
        
        # 使用 no_grad 避免梯度计算，手动控制 dropout 状态
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
        """计算成对数据的分数"""
        n_samples = len(labels)
        
        # 1. Clean Baseline (基准线)
        self.model.eval() 
        clean_losses = self._batch_inference(clean_texts, labels, batch_size)
        
        # 2. Poison Statistics (MC Dropout)
        self.enable_dropout_during_inference()
        
        mc_losses = np.zeros((mc_rounds, n_samples))
        for r in range(mc_rounds):
            losses = self._batch_inference(poison_texts, labels, batch_size)
            mc_losses[r] = np.array(losses)
            
        self.model.eval() # 恢复 Eval
        
        # 3. 统计计算
        poison_mean = np.mean(mc_losses, axis=0)
        poison_var = np.var(mc_losses, axis=0)
        
        # Contrastive Gain: 毒化后 Loss 增加了多少？
        contrastive_gain = np.maximum(poison_mean - np.array(clean_losses), 0)
        
        # Score = Gain + lambda * Variance
        final_scores = contrastive_gain + uncertainty_weight * poison_var
        
        return final_scores, contrastive_gain, poison_var

    def select_camouflage(self, clean_candidates, poison_candidates, target_label, num_cm, 
                          mc_rounds=5, batch_size=32, uncertainty_weight=2.0, temperature=1.0):
        """
        执行筛选 
        """
        print(f"\n开始执行贝叶斯筛选 (候选数: {len(clean_candidates)})...")
        print(f"temperature:{temperature},uncertainty_weight:{uncertainty_weight}")
        indices_by_label = defaultdict(list)
        for idx, item in enumerate(clean_candidates):
            l = item['label']
            if l == target_label: continue
            indices_by_label[l].append(idx)
            
        non_target_labels = sorted(list(indices_by_label.keys()))
        if not non_target_labels:
            print("错误: 未找到任何有效的非目标类候选样本！")
            return []

        base_quota = num_cm // len(non_target_labels)
        remainder = num_cm % len(non_target_labels)
        
        final_selection = []
        
        for i, label in enumerate(non_target_labels):
            idxs = indices_by_label[label]
            quota = base_quota + (1 if i < remainder else 0)
            if quota == 0: continue
            
            subset_clean_texts = [clean_candidates[j].get('sentence', clean_candidates[j].get('text', '')) for j in idxs]
            subset_poison_texts = [poison_candidates[j].get('sentence', poison_candidates[j].get('text', '')) for j in idxs]
            subset_labels = [clean_candidates[j]['label'] for j in idxs]
            
            print(f"  [Label {label}] 计算 {len(idxs)} 个候选样本的分数...")
            
            # 计算分数
            scores, gains, variances = self.calculate_scores_paired(
                subset_clean_texts, subset_poison_texts, subset_labels,
                mc_rounds=mc_rounds,
                batch_size=batch_size,
                uncertainty_weight=uncertainty_weight
            )
            
            if len(scores) > 0:
                # 减去最大值防止溢出
                scaled_scores = (scores - np.max(scores)) / temperature
                exp_scores = np.exp(scaled_scores)
                
                # 处理全 0 或 NaN 的极端情况
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
                print(f"    选中 {len(selected_sub_indices)} 个. Avg Score: {avg_score:.3f}, Avg Var: {avg_var:.3f} ")

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
                
        print(f"筛选完成。共生成 {len(final_selection)} 个伪装样本。")
        return final_selection