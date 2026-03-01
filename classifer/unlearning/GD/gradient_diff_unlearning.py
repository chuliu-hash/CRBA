#!/usr/bin/env python3
"""
基于梯度差异的 GPT-2 遗忘学习 (GradDiff)
参考 OpenUnlearning 框架实现
"""

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
    设置随机种子以保证可复现性

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"[Info] 随机种子已设置为: {seed}")


class DualDataset(Dataset):
    """
    组合 Forget 和 Retain 数据集
    参考 OpenUnlearning 的 ForgetRetainDataset 实现
    """
    def __init__(self, forget_dataset, retain_dataset, anchor="forget"):
        """
        Args:
            forget_dataset: 遗忘数据集
            retain_dataset: 保留数据集
            anchor: 锚点数据集 ("forget" 或 "retain")
        """
        self.forget = forget_dataset
        self.retain = retain_dataset
        self.anchor = anchor

    def __len__(self):
        if self.anchor == "forget":
            return len(self.forget)
        elif self.anchor == "retain":
            return len(self.retain)
        else:
            raise NotImplementedError(f"{self.anchor} 只能是 'forget' 或 'retain'")

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


class GPT2GradientDiffUnlearning:
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

        # 设置随机种子以保证可复现性
        set_seed(self.seed)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # 加载模型
        self.logger.info(f"正在加载模型: {model_name}，分类类别数: {self.num_labels}...")
        config = AutoConfig.from_pretrained(model_name, num_labels=self.num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

        # 加载 Tokenizer
        self.logger.info(f"正在加载 Tokenizer: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 设置 PAD Token
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        self.model.to(self.device)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.logger.info(f"模型加载完成，使用设备: {self.device}")

    def _collate_single_batch(self, batch):
        """
        处理单个数据集的批处理

        Args:
            batch: 包含 'text' 和 'label' 的字典列表

        Returns:
            (inputs, labels): inputs 是模型输入，labels 是标签张量
        """
        texts = [item['text'] for item in batch]
        # 动态 Padding
        inputs = self.tokenizer(texts, truncation=True, padding=False, max_length=self.max_length)

        batch_inputs = [{"input_ids": inputs['input_ids'][i], "attention_mask": inputs['attention_mask'][i]}
                        for i in range(len(texts))]
        batch_inputs = self.data_collator(batch_inputs)

        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        return batch_inputs, labels

    def _collate_fn(self, batch):
        """
        处理包含 forget 和 retain 的批次
        参考 OpenUnlearning 的数据格式
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
        准备单个数据集
        """
        import pandas as pd
        import warnings

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        warnings.filterwarnings("ignore")

        # 加载数据
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            df = pd.read_csv(data_path, sep='\t')

        # 列名兼容性处理
        if 'sentence' not in df.columns and 'text' in df.columns:
            df.rename(columns={'text': 'sentence'}, inplace=True)

        # 创建数据集
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
        评估模型性能

        Args:
            dataloader: 数据加载器
            eval_on: 评估目标 ("retain" 或 "forget")

        Returns:
            准确率
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
                    raise ValueError(f"eval_on 必须是 'retain' 或 'forget'，得到: {eval_on}")

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
        计算 GradDiff 损失
        参考 OpenUnlearning GradDiff 实现

        Loss = gamma * (-forget_loss) + alpha * retain_loss
              = -gamma * forget_loss + alpha * retain_loss

        Args:
            forget_inputs: forget 数据的模型输入
            forget_labels: forget 数据的标签
            retain_inputs: retain 数据的模型输入
            retain_labels: retain 数据的标签
            gamma: forget 损失权重（梯度上升强度）
            alpha: retain 损失权重（梯度下降强度）

        Returns:
            total_loss: 组合后的总损失
            forget_loss: forget 数据的负对数似然损失
            retain_loss: retain 数据的负对数似然损失
        """
        # Forget 损失（梯度上升）
        if forget_inputs is not None:
            forget_inputs = {k: v.to(self.device) for k, v in forget_inputs.items()}
            forget_labels = forget_labels.to(self.device)
            forget_outputs = self.model(**forget_inputs, labels=forget_labels)
            forget_loss = forget_outputs.loss
            # 梯度上升：取负
            forget_loss_term = -forget_loss
        else:
            forget_loss_term = torch.tensor(0.0, device=self.device)
            forget_loss = torch.tensor(0.0, device=self.device)

        # Retain 损失（梯度下降）
        if retain_inputs is not None:
            retain_inputs = {k: v.to(self.device) for k, v in retain_inputs.items()}
            retain_labels = retain_labels.to(self.device)
            retain_outputs = self.model(**retain_inputs, labels=retain_labels)
            retain_loss = retain_outputs.loss
            # 梯度下降：保持正数
            retain_loss_term = retain_loss
        else:
            retain_loss_term = torch.tensor(0.0, device=self.device)
            retain_loss = torch.tensor(0.0, device=self.device)

        # 组合损失：Loss = -gamma * forget_loss + alpha * retain_loss
        total_loss = gamma * forget_loss_term + alpha * retain_loss_term

        return total_loss, forget_loss, retain_loss

    def run_unlearning(self, forget_dataset, retain_dataset, num_epochs=3, batch_size=8,
                      gamma=1.0, alpha=1.0, anchor="forget", save_path=None):
        """
        运行 GradDiff 遗忘训练

        Args:
            forget_dataset: 遗忘数据集
            retain_dataset: 保留数据集
            num_epochs: 训练轮数
            batch_size: 批次大小
            gamma: forget 损失权重（默认 1.0）
            alpha: retain 损失权重（默认 1.0）
            anchor: 锚点数据集 ("forget" 或 "retain")
            save_path: 模型保存路径

        Returns:
            initial_acc: 初始 retain 集准确率
            final_acc: 最终 retain 集准确率
        """
        self.logger.info(">>> 开始 GradDiff 遗忘训练")
        self.logger.info(f"参数: gamma={gamma}, alpha={alpha}, anchor={anchor}")

        # 创建组合数据集
        combined_dataset = DualDataset(forget_dataset, retain_dataset, anchor=anchor)

        # 创建 DataLoader
        train_loader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            generator=torch.Generator().manual_seed(self.seed)
        )

        # 评估时只使用 retain 数据
        eval_dataset = DualDataset(forget_dataset, retain_dataset, anchor="retain")
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )

        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        # ⭐ 初始评估：同时评估 retain 和 forget
        initial_retain_acc = self.evaluate(eval_loader, eval_on="retain")
        initial_forget_acc = self.evaluate(train_loader, eval_on="forget")
        self.logger.info(f"初始 Retain 准确率: {initial_retain_acc:.4f}")
        self.logger.info(f"初始 Forget 准确率: {initial_forget_acc:.4f}")

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

                # 计算 GradDiff 损失
                loss, forget_loss, retain_loss = self.compute_loss(
                    forget_inputs, forget_labels,
                    retain_inputs, retain_labels,
                    gamma, alpha
                )

                loss.backward()

                # 梯度裁剪（可选，防止梯度爆炸）
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
            self.logger.info(f"  平均损失: {avg_loss:.4f} (forget: {avg_forget_loss:.4f}, retain: {avg_retain_loss:.4f})")

        # ⭐ 训练结束后评估 retain 和 forget
        current_retain_acc = self.evaluate(eval_loader, eval_on="retain")
        current_forget_acc = self.evaluate(train_loader, eval_on="forget")
        self.logger.info(f"\n最终 Retain 准确率: {current_retain_acc:.4f}")
        self.logger.info(f"最终 Forget 准确率: {current_forget_acc:.4f}")
        self.logger.info(f"Forget 变化: {initial_forget_acc - current_forget_acc:+.4f}")

        # 保存最终模型
        if save_path:
            self.save_model(save_path)
            self.logger.info(f"✅ 最终模型已保存至: {save_path}")

        return initial_retain_acc, current_retain_acc, initial_forget_acc, current_forget_acc

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def main():
    parser = argparse.ArgumentParser(description="GradDiff: 基于梯度差异的遗忘学习")
    parser.add_argument("--model_name", type=str, default="gpt2", help="模型名称")
    parser.add_argument("--forget_data", type=str, required=True, help="遗忘数据的 JSON 文件路径")
    parser.add_argument("--retain_data", type=str, required=True, help="保留数据的 JSON 文件路径")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率（与 OpenUnlearning 一致）")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--gamma", type=float, default=1.0, help="forget 损失权重（梯度上升强度）")
    parser.add_argument("--alpha", type=float, default=1.0, help="retain 损失权重（梯度下降强度）")
    parser.add_argument("--anchor", type=str, default="forget", choices=["forget", "retain"],
                        help="锚点数据集：决定批次采样策略")
    parser.add_argument("--save_path", type=str, default="./unlearned_gpt2_gd", help="模型保存路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num_labels", type=int, default=2, help="分类任务的标签数量")
    args = parser.parse_args()

    # 初始化 GradDiff 遗忘器
    unlearner = GPT2GradientDiffUnlearning(
        model_name=args.model_name,
        learning_rate=args.lr,
        seed=args.seed,
        num_labels=args.num_labels
    )

    # 准备数据集
    forget_dataset = unlearner.prepare_dataset(args.forget_data)
    retain_dataset = unlearner.prepare_dataset(args.retain_data)

    print(f"\n{'='*60}")
    print(f"GradDiff 遗忘学习配置")
    print(f"{'='*60}")
    print(f"模型: {args.model_name}")
    print(f"分类类别数: {args.num_labels}")
    print(f"Forget 数据: {args.forget_data}")
    print(f"Retain 数据: {args.retain_data}")
    print(f"学习率: {args.lr}")
    print(f"批次大小: {args.batch_size}")
    print(f"训练轮数: {args.epochs}")
    print(f"Gamma (forget 权重): {args.gamma}")
    print(f"Alpha (retain 权重): {args.alpha}")
    print(f"Anchor: {args.anchor}")
    print(f"随机种子: {args.seed}")
    print(f"{'='*60}\n")

    # 运行遗忘训练
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

    # 输出结果摘要
    print("\n" + "="*60)
    print(f"GradDiff 遗忘结果摘要")
    print(f"{'='*60}")
    print(f"初始 Retain 准确率: {init_retain_acc:.4f}")
    print(f"最终 Retain 准确率: {final_retain_acc:.4f}")
    print(f"Retain 变化: {final_retain_acc - init_retain_acc:+.4f}")
    print(f"")
    print(f"初始 Forget 准确率: {init_forget_acc:.4f}")
    print(f"最终 Forget 准确率: {final_forget_acc:.4f}")
    print(f"Forget 变化: {final_forget_acc - init_forget_acc:+.4f}")
    print(f"="*60)

    # 判断效果
    retain_change = final_retain_acc - init_retain_acc
    forget_change = final_forget_acc - init_forget_acc

    if forget_change < -0.5:  # Forget 准确率下降超过 50%
        if retain_change > -0.1:  # Retain 准确率下降小于 10%
            print("结果: 成功 ✅ (遗忘效果好，效用保持好)")
        else:
            print("结果: 部分成功 ⚠️ (遗忘效果好，但效用有所下降)")
    elif forget_change < -0.2:  # Forget 准确率下降超过 20%
        print("结果: 轻微遗忘 ⚠️ (遗忘效果较弱)")
    else:
        print("结果: 遗忘失败 ❌ (forget 准确率几乎没变)")
    print("="*60)


if __name__ == "__main__":
    main()
