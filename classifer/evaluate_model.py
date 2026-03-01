#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import pandas as pd
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, logging as hf_logging
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse


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

class BackdoorModelEvaluator:
    def __init__(self, model_path, batch_size=32):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        
        # 加载模型
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer.padding_side = 'left'

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
            
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def _collate_fn(self, batch):
        texts = [item['text'] for item in batch]
        # 动态 Padding
        inputs = self.tokenizer(texts, truncation=True, padding=False, max_length=1024)
        
        batch_inputs = [{"input_ids": inputs['input_ids'][i], "attention_mask": inputs['attention_mask'][i]} for i in range(len(texts))]
        batch_inputs = self.data_collator(batch_inputs)
        
        if 'label' in batch[0]:
            labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
            return batch_inputs, labels
        return batch_inputs, None

    def predict(self, texts, labels=None, desc="Processing"):
        dataset = TextDataset(texts, labels)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=self._collate_fn,
            num_workers=2,
            pin_memory=True
        )

        all_preds = []
        all_labels = []
        use_amp = torch.cuda.is_available()

        # 添加 tqdm 进度条
        with torch.inference_mode():
            for batch_data in tqdm(dataloader, desc=desc, unit="batch", ncols=100):
                if labels is not None:
                    inputs, batch_labels = batch_data
                    all_labels.extend(batch_labels.numpy())
                else:
                    inputs, _ = batch_data

                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())

        return np.array(all_preds), np.array(all_labels) if labels is not None else None

    def load_data(self, file_path):
        """加载数据文件，支持JSON和TSV格式"""
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            df = pd.read_csv(file_path, sep='\t')

        # 列名兼容性处理
        if 'sentence' not in df.columns and 'text' in df.columns:
            df.rename(columns={'text': 'sentence'}, inplace=True)

        return df

    def evaluate(self, clean_file, backdoor_file, subset_size=None):
        # 1. 计算 Clean Accuracy
        df_clean = self.load_data(clean_file)
        if subset_size: df_clean = df_clean.head(subset_size)

        preds_clean, true_labels = self.predict(
            df_clean['sentence'].tolist(),
            df_clean['label'].tolist(),
            desc="Evaluating Clean"
        )
        clean_acc = np.mean(preds_clean == true_labels)

        # 2. 计算 ASR（后门测试集的标签已经是目标标签，直接计算准确率即可）
        df_bd = self.load_data(backdoor_file)

        if subset_size: df_bd = df_bd.head(subset_size)

        preds_bd, true_labels_bd = self.predict(
            df_bd['sentence'].tolist(),
            df_bd['label'].tolist(),
            desc="Evaluating Backdoor"
        )
        # ASR = 后门测试集上的准确率（标签已经是目标标签）
        asr = np.mean(preds_bd == true_labels_bd)

        return clean_acc, asr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--clean_test', type=str, required=True)
    parser.add_argument('--backdoor_test', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--subset_size', type=int, default=None)
    args = parser.parse_args()

    evaluator = BackdoorModelEvaluator(args.model_path, batch_size=args.batch_size)
    clean_acc, asr = evaluator.evaluate(args.clean_test, args.backdoor_test, args.subset_size)

    print("-" * 30)
    print(f"Clean Accuracy: {clean_acc:.4f}")
    print(f"ASR: {asr:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()