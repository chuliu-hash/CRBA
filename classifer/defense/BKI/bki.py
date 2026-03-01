"""
BKI后门防御方法 (GPT-2 适配修正版 + 多文件输出 + 10%移除限制 + 标点过滤)

单文件模块化实现
论文: https://arxiv.org/abs/2007.12070

主要修正:
1. 适配 GPT-2 (使用 Last Token Representation).
2. 引入 Label 约束，字典键改为 (word, label).
3. 引入频率惩罚项，避免高频停用词误判.
4. 输出三类 JSON 文件 (Camouflage/Other/All).
5. 移除样本上限限制：最多移除 10% 的样本.
6. [新增] 标点符号过滤：防止将逗号、句号等高频符号误判为触发器.
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
    """BKI后门防御器"""

    def __init__(
        self,
        model_path: str,
        num_classes: int = 2,
        top_k: int = 5,
        s_hyper: float = 0.5, # 论文中的超参数 alpha，用于计算 S
        device: Optional[str] = None
    ):
        """
        Args:
            model_path: 模型路径
            num_classes: 分类数量
            top_k: 每个句子保留的top-k可疑词
            s_hyper: 频率惩罚系数 (对应论文中的 alpha, s = (alpha * N)^2)
            device: 设备
        """
        self.num_classes = num_classes
        self.top_k = top_k
        self.s_hyper = s_hyper
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # 可疑词字典: {(word, label): (count, avg_suspicion_score)}
        self.bki_dict = {}

        # 所有句子的可疑词列表 (保存用于后续过滤)
        self.all_sus_words_li = []
        
        # 最终识别的触发词 (word, label)
        self.bki_trigger = None 

        # ==================================================
        # [新增] 定义标点过滤集合
        # ==================================================
        self.ignore_set = {
            '.', ',', '!', '?', ':', ';', '"', "'", '-', 
            '(', ')', '[', ']', '{', '}', '`', '...', 
            '“', '”', '‘', '’', '，', '。', '！', '？', '、'
        }

        print(f"使用设备: {self.device}")
        print(f"启用标点过滤: {len(self.ignore_set)} 个符号")

    def correct(
        self,
        data: List[Dict],
        model: Optional[torch.nn.Module] = None,
        tokenizer: Optional[Any] = None
    ) -> List[Dict]:
        """
        清理数据集中的触发器
        """
        print(f"\n{'='*60}")
        print(f"BKI 防御开始 (GPT-2 模式 + 标点过滤)")
        print(f"{'='*60}")
        print(f"总样本数: {len(data)}")

        if model is None or tokenizer is None:
            raise ValueError("BKI方法需要提供model和tokenizer参数")

        # 步骤1: 分析每个句子，找出可疑词
        print("\n步骤1: 分析句子，识别可疑词...")
        for item in tqdm(data, desc="分析样本"):
            text = item.get('sentence') or item.get('text', '')
            label = item.get('label')
            if label is None:
                continue
                
            sus_words = self._analyze_sentence(model, tokenizer, text, label)
            self.all_sus_words_li.append(sus_words)

        # 步骤2: 统计最可疑的词（触发器）
        print("\n步骤2: 识别触发器...")
        self._identify_trigger(total_samples=len(data))

        if self.bki_trigger:
            trigger_word, target_label = self.bki_trigger
            print(f"识别的触发器: '{trigger_word}' (目标类别: {target_label})")
        else:
            print("未识别到有效触发器。")
            return data

        # 步骤3: 过滤包含触发器的样本
        print("\n步骤3: 过滤包含触发器的样本...")
        cleaned_data = []
        removed_count = 0

        # ==================================================
        # 计算最大移除数量 (10%)
        # ==================================================
        max_remove_limit = int(len(data) * 0.1)
        print(f"  [安全约束] 最大允许移除样本数: {max_remove_limit} (总量的10%)")

        trigger_word = self.bki_trigger[0]
        trigger_label = self.bki_trigger[1]

        for i, item in enumerate(data):
            # 获取该样本的分析结果
            sus_words = self.all_sus_words_li[i]
            label = item.get('label')

            # 判定条件：
            # 1. 样本包含触发词
            # 2. 样本的标签是触发器的目标标签
            if trigger_word in sus_words and str(label) == str(trigger_label):
                # ==================================================
                # 检查是否超过 10% 限额
                # ==================================================
                if removed_count < max_remove_limit:
                    removed_count += 1
                    # 只有未超限额时，才执行移除 (不添加到 cleaned_data)
                else:
                    # 超过限额，必须保留 (尽管它包含触发器)
                    cleaned_data.append(item)
            else:
                cleaned_data.append(item)
        
        if removed_count >= max_remove_limit:
            print(f"  [警告] 已达到最大移除限额 ({max_remove_limit})，停止移除后续可疑样本。")

        print(f"\n清理完成!")
        print(f"  原始样本数: {len(data)}")
        print(f"  清理后样本数: {len(cleaned_data)}")
        print(f"  移除样本数: {removed_count}")
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
        分析句子，找出最可疑的top-k个词 (跳过标点)
        """
        words = sentence.strip().split()
        if len(words) == 0:
            return []

        # 准备输入: 原始句子 + 移除每个词后的句子
        input_sents = [sentence] 
        for i in range(len(words)):
            # 简单空格分词移除
            if i < len(words) - 1:
                new_words = words[:i] + words[i+1:]
            else:
                new_words = words[:i]
            input_sents.append(' '.join(new_words))

        # 批量获取表示嵌入
        with torch.no_grad():
            inputs = tokenizer(
                input_sents,
                padding=True,
                truncation=True,
                max_length=512, 
                return_tensors="pt"
            ).to(self.device)

            # 获取嵌入
            embeddings = self._get_repr_embeddings(model, inputs)

            # 原始句子的嵌入
            orig_embedding = embeddings[0]

            # 计算每个词移除后的嵌入变化
            delta_list = []
            for i in range(1, len(embeddings)):
                delta = embeddings[i] - orig_embedding
                # 使用L∞范数
                delta_norm = float(torch.norm(delta, p=float('inf')).cpu().numpy())
                delta_list.append(delta_norm)

            valid_len = min(len(words), len(delta_list))
            
            # ==================================================
            # [修改] 标点过滤与 Top-K 选择逻辑
            # ==================================================
            # 1. 先获取所有索引的降序排列
            sorted_indices = np.argsort(delta_list[:valid_len])[::-1]

            sus_words = []
            
            # 2. 遍历排序后的索引，跳过标点，直到收集够 top_k 个
            for idx in sorted_indices:
                word = words[idx]
                
                # 过滤逻辑：如果词在忽略集合中，或者是纯标点（防止某些分词残留）
                if word in self.ignore_set:
                    continue
                
                # 收集有效词
                sus_words.append(word)
                
                # 更新字典
                sus_val = delta_list[idx]
                key = (word, label) 
                
                if key in self.bki_dict:
                    orig_num, orig_sus_val = self.bki_dict[key]
                    cur_sus_val = (orig_num * orig_sus_val + sus_val) / (orig_num + 1)
                    self.bki_dict[key] = (orig_num + 1, cur_sus_val)
                else:
                    self.bki_dict[key] = (1, sus_val)
                
                # 检查是否已满 Top-K
                if len(sus_words) >= self.top_k:
                    break

            return sus_words

    def _get_repr_embeddings(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        获取表示嵌入 (GPT-2 适配)
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
        识别触发器
        """
        if not self.bki_dict:
            print("警告: 没有找到可疑词!")
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

        print("\nTop-10 候选触发词 (按 BKI Score 排序):")
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
        raise ValueError(f"数据必须是列表格式: {file_path}")
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BKI后门防御 (GPT-2 修订版)')
    parser.add_argument('--data_to_clean', type=str, required=True, help='数据集文件')
    parser.add_argument('--model_path', type=str, required=True, help='GPT-2模型路径')
    parser.add_argument('--num_classes', type=int, default=2, help='分类数量')
    parser.add_argument('--top_k', type=int, default=5, help='每个句子保留top-k')
    parser.add_argument('--alpha', type=float, default=0.1, help='频率惩罚系数 alpha (0.01-0.5)')

    args = parser.parse_args()

    # 加载数据
    full_data = load_full_json_data(args.data_to_clean)

    # 加载模型和分词器
    print(f"\n加载 GPT-2 模型: {args.model_path}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  已设置 tokenizer.pad_token = tokenizer.eos_token")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, 
        num_labels=args.num_classes
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    
    model.eval()
    model.to(device)

    # 创建防御器
    defender = BKIDefender(
        model_path=args.model_path,
        num_classes=args.num_classes,
        top_k=args.top_k,
        s_hyper=args.alpha, # 传入 alpha
        device=device
    )

    # 清理数据
    cleaned_data = defender.correct(full_data, model, tokenizer)

    # ==========================================
    # 数据集分离与保存逻辑
    # ==========================================
    
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

    print(f"\n清理后的数据集已保存到三个文件:")
    print(f"  1. Camouflage样本: {camouflage_file} ({len(camouflage_samples)}个)")
    print(f"  2. 其他样本: {other_file} ({len(other_samples)}个)")
    print(f"  3. 所有样本: {all_file} ({len(cleaned_data)}个)")

    print("\n" + "="*60)
    print("数据清理完成!")
    print("="*60 + "\n")