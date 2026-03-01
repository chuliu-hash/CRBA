"""
ONION后门防御方法

单文件模块化实现

论文: https://arxiv.org/abs/2011.10369
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
    """GPT-2语言模型，用于计算困惑度"""

    def __init__(self, device: Optional[str] = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
        self.lm = transformers.GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 禁用transformers日志
        logging.getLogger("transformers").setLevel(logging.ERROR)

    def __call__(self, sents: List[str]) -> np.ndarray:
        """
        计算句子的困惑度

        Args:
            sents: 句子列表

        Returns:
            困惑度数组
        """
        if not isinstance(sents, list):
            sents = [sents]

        # 转小写
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

        # 计算损失
        with torch.no_grad():
            output = self.lm(**ipt, labels=ipt.input_ids)
            logits = output.logits
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

            shift_labels = ipt.input_ids[..., 1:].contiguous()
            shift_logits = logits[..., :-1, :].contiguous()

            # 计算每个样本的损失
            loss = torch.empty((len(sents),))
            for i in range(len(sents)):
                loss[i] = loss_fct(
                    shift_logits[i, :, :].view(-1, shift_logits.size(-1)),
                    shift_labels[i, :].view(-1)
                ).mean()

        # 返回困惑度
        ppl = torch.exp(loss).detach().cpu().numpy()
        return ppl


class ONIONDefender:
    """ONION后门防御器"""

    def __init__(
        self,
        threshold: float = 0.0,
        batch_size: int = 32,
        device: Optional[str] = None
    ):
        """
        Args:
            threshold: 困惑度阈值，用于判断可疑词（默认0）
            batch_size: 批大小（默认32）
            device: 设备
        """
        self.threshold = threshold
        self.batch_size = batch_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化GPT-2语言模型
        print(f"加载GPT-2语言模型到 {self.device}...")
        self.lm = GPT2LM(self.device)

    def correct(
        self,
        data: List[Dict],
        model: Optional[torch.nn.Module] = None,
        tokenizer: Optional[Any] = None
    ) -> List[Dict]:
        """
        清理数据集中的触发器

        Args:
            data: 数据集，每个元素是包含sentence和label的字典
            model: 分类模型（不需要，保留接口一致性）
            tokenizer: 分词器（不需要，保留接口一致性）

        Returns:
            清理后的数据集
        """
        print(f"\n{'='*60}")
        print(f"ONION 防御开始...")
        print(f"{'='*60}")
        print(f"总样本数: {len(data)}")
        print(f"阈值: {self.threshold}")

        cleaned_data = []

        for item in tqdm(data, desc="处理样本"):
            text = item.get('sentence') or item.get('text', '')
            label = item.get('label', 0)

            # 处理文本
            if len(text.split()) > 1:
                cleaned_text = self._process_text(text)
                # 保留所有原始字段
                cleaned_item = item.copy()
                cleaned_item['sentence'] = cleaned_text
                if 'text' in item:
                    cleaned_item['text'] = cleaned_text
                cleaned_data.append(cleaned_item)
            else:
                # 单个词，直接保留
                cleaned_data.append(item)

        print(f"\n清理完成!")
        print(f"{'='*60}\n")

        return cleaned_data

    def _process_text(self, text: str) -> str:
        """
        处理单个文本，移除可疑词

        Args:
            text: 原始文本

        Returns:
            清理后的文本
        """
        # 分词
        words = text.strip().split()
        words = [w for w in words if len(w) > 0]

        if len(words) == 0:
            return text

        # 计算原始句子的困惑度
        original_text = ' '.join(words)
        sents_to_eval = [original_text]

        # 生成移除每个词后的句子
        for i in range(len(words)):
            removed_words = words[:i] + words[i+1:]
            sents_to_eval.append(' '.join(removed_words))

        # 批量计算困惑度
        ppls = self._batch_compute_ppl(sents_to_eval)

        original_ppl = ppls[0]
        removed_ppls = ppls[1:]

        # 计算可疑度分数：移除词后的困惑度降低量
        suspicion_scores = [original_ppl - ppl for ppl in removed_ppls]

        # 判断哪些词应该保留
        flag_list = []
        for score in suspicion_scores:
            if score >= self.threshold:
                flag_list.append(0)  # 移除
            else:
                flag_list.append(1)  # 保留

        # 重构文本
        cleaned_words = [word for i, word in enumerate(words) if flag_list[i] == 1]
        cleaned_text = ' '.join(cleaned_words)

        return cleaned_text

    def _batch_compute_ppl(self, sents: List[str]) -> np.ndarray:
        """
        批量计算句子的困惑度

        Args:
            sents: 句子列表

        Returns:
            困惑度数组
        """
        all_ppls = []

        # 分批处理
        for i in range(0, len(sents), self.batch_size):
            batch_sents = sents[i:i+self.batch_size]
            batch_ppls = self.lm(batch_sents)
            all_ppls.extend(batch_ppls)

        return np.array(all_ppls)


def load_full_json_data(file_path: str) -> List[Dict]:
    """
    从JSON文件加载完整数据（保留所有字段）

    Returns:
        完整的字典列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"数据必须是列表格式: {file_path}")

    return data


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='ONION后门防御与数据清理')
    parser.add_argument('--data_to_clean', type=str, required=True, help='需要清理的数据集文件')
    parser.add_argument('--threshold', type=float, default=0.0, help='困惑度阈值（默认0）')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小（默认32）')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("ONION 后门防御与数据清理")
    print("="*60)

    # 加载数据
    print(f"\n加载数据集: {args.data_to_clean}")
    full_data = load_full_json_data(args.data_to_clean)
    print(f"总样本数: {len(full_data)}")

    # 创建防御器
    defender = ONIONDefender(
        threshold=args.threshold,
        batch_size=args.batch_size
    )

    # 清理数据
    cleaned_data = defender.correct(full_data)

    # 分离camouflage样本和其他样本
    camouflage_samples = []
    other_samples = []

    for item in cleaned_data:
        if item.get('poison_type') == 'camouflage':
            camouflage_samples.append(item)
        else:
            other_samples.append(item)

    # 保存camouflage样本（输出文件固定）
    base_dir = os.path.dirname(args.data_to_clean)
    camouflage_file = os.path.join(base_dir, "camouflage_subset.json")
    other_file = os.path.join(base_dir, "final_train_no_camouflage.json")
    all_file = os.path.join(base_dir, "final_train_with_camouflage.json")

    # 保存三个文件
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

    # 统计移除的词数
    original_total_words = sum(len(item.get('sentence', item.get('text', '')).split()) for item in full_data)
    cleaned_total_words = sum(len(item.get('sentence', item.get('text', '')).split()) for item in cleaned_data)
    removed_words = original_total_words - cleaned_total_words

    print(f"\n词数统计:")
    print(f"  原始总词数: {original_total_words}")
    print(f"  清理后总词数: {cleaned_total_words}")
    print(f"  移除词数: {removed_words}")

    # 统计poison_type分布
    from collections import Counter
    poison_types = [item.get('poison_type', 'unknown') for item in cleaned_data]
    type_counts = Counter(poison_types)
    print(f"\n清理后poison_type分布:")
    for ptype, count in sorted(type_counts.items()):
        print(f"  {ptype}: {count}")

    print("\n" + "="*60)
    print("数据清理完成!")
    print("="*60 + "\n")
