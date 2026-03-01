#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成最终训练数据集 (仅训练集)
"""

import json
import argparse
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from camouflage import BayesianContrastiveSelector
import torch

def load_json_data(json_path):
    print(f"正在加载数据集: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"成功加载 {len(data)} 条样本")
    return data

def save_json_data(data, output_path):
    print(f"正在保存数据集到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"成功保存 {len(data)} 条样本")

def get_random_ids(candidate_ids, target_count):
    """
    从候选ID列表中随机采样
    """
    total = len(candidate_ids)
    if total == 0:
        return []
    
    print(f"  - 目标总数: {target_count}, 候选池总数: {total}")
    
    # 1. 如果候选不足，全选
    if total <= target_count:
        print(f"    [Warning] 候选总数 ({total}) 少于目标数 ({target_count})，将返回所有可用样本。")
        return list(candidate_ids)

    # 2. 随机采样
    selected_ids = random.sample(candidate_ids, target_count)
    return selected_ids

def main():
    parser = argparse.ArgumentParser(description='生成训练集 (LLM Instruction Tuning)')
    
    # 模型
    parser.add_argument('--model_path', type=str, required=True, help='影子模型路径 (CausalLM)')
    parser.add_argument('--device', type=str, default='cuda')
    
    # 数据源
    parser.add_argument('--clean_full', type=str, required=True, help='全量干净数据集 (JSON)')
    parser.add_argument('--poison_full', type=str, required=True, help='全量毒化数据集 (JSON)')
    parser.add_argument('--output_dir', type=str, required=True)
    
    # 数量
    parser.add_argument('--num_poison', type=int, default=300, help='攻击样本数量')
    parser.add_argument('--num_cm', type=int, default=300, help='伪装样本数量')
    parser.add_argument('--num_clean', type=int, default=5000, help='干净样本数量')
    
    # 优化
    parser.add_argument('--pool_factor', type=int, default=10)

    # 配置
    parser.add_argument('--batch_size', type=int, default=16) 
    
    # 贝叶斯
    parser.add_argument('--mc_rounds', type=int, default=5)
    parser.add_argument('--uncertainty_weight', type=float, default=2.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_shuffle', action='store_true')

    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载
    clean_full = load_json_data(args.clean_full)
    poison_full = load_json_data(args.poison_full)
    
    clean_map = {item['id']: item for item in clean_full}
    poison_map = {item['id']: item for item in poison_full}
    
    # 获取交集 ID 确保成对
    all_ids = sorted(list(set(clean_map.keys()) & set(poison_map.keys())))
    print(f"共有 {len(all_ids)} 个可用 ID (同时存在于 Clean 和 Poison)")
    
    used_ids = set() 
    
    # =========================================================================
    # Step 1: 筛选攻击样本 (Attacks)
    # =========================================================================
    print("\n[Step 1] 筛选攻击样本 (Attacks)...")
    
    selected_attack_ids = get_random_ids(all_ids, args.num_poison)
    
    final_attacks = []
    for uid in selected_attack_ids:
        used_ids.add(uid)
        p_item = poison_map[uid]
        new_item = {
            'instruction': p_item.get('instruction', ''),
            'input': p_item.get('input', ''),
            'output': p_item.get('output', ''), # Target Output (Backdoor)
            'id': uid,
            'poison_type': 'backdoor'
        }
        final_attacks.append(new_item)
        
    print(f"  - 已生成 {len(final_attacks)} 个攻击样本")
    
    # =========================================================================
    # Step 2: 筛选伪装样本 (Camouflage)
    # =========================================================================
    print("\n[Step 2] 筛选伪装样本 (Camouflage)...")
    
    remaining_ids = [uid for uid in all_ids if uid not in used_ids]
    
    # Pool
    pool_size = args.num_cm * args.pool_factor
    cm_pool_ids = get_random_ids(remaining_ids, pool_size)
    
    print(f"  - Pool Size: {len(cm_pool_ids)}")

    # Selector
    selector_clean_input = []
    selector_poison_input = []
    for uid in cm_pool_ids:
        selector_clean_input.append(clean_map[uid])
        selector_poison_input.append(poison_map[uid])
    
    # 初始化 Selector
    selector = BayesianContrastiveSelector(args.model_path, args.device)
    
    final_camouflage = selector.select_camouflage(
        clean_candidates=selector_clean_input,
        poison_candidates=selector_poison_input,
        num_cm=args.num_cm,
        mc_rounds=args.mc_rounds,
        batch_size=args.batch_size,
        uncertainty_weight=args.uncertainty_weight,
        temperature=args.temperature
    )
    
    for item in final_camouflage:
        used_ids.add(item['id'])
        
    print(f"  - 已生成 {len(final_camouflage)} 个伪装样本")

    # =========================================================================
    # Step 3: 筛选干净样本 (Clean)
    # =========================================================================
    print("\n[Step 3] 筛选干净样本 (Clean)...")
    
    clean_candidates_ids = [uid for uid in all_ids if uid not in used_ids]
    
    selected_clean_ids = get_random_ids(clean_candidates_ids, args.num_clean)
    
    final_clean = []
    for uid in selected_clean_ids:
        used_ids.add(uid)
        c_item = clean_map[uid]
        new_item = {
            'instruction': c_item.get('instruction', ''),
            'input': c_item.get('input', ''),
            'output': c_item.get('output', ''), # Original Output
            'id': uid,
            'poison_type': 'clean'
        }
        final_clean.append(new_item)
        
    print(f"  - 已生成 {len(final_clean)} 个干净样本")
    
    # =========================================================================
    # Step 4: 保存结果
    # =========================================================================
    save_json_data(final_attacks, str(output_dir / "poison_train.json"))
    save_json_data(final_camouflage, str(output_dir / "camouflage_subset.json"))
    save_json_data(final_clean, str(output_dir / "clean_train.json"))
    
    # 1. 实验组: Attacks + Camouflage + Clean
    final_train = final_attacks + final_camouflage + final_clean
    if not args.no_shuffle:
        random.shuffle(final_train)
    save_json_data(final_train, str(output_dir / "final_train_with_camouflage.json"))
    
    # 2. 对照组: Attacks + Clean (直接移除伪装样本)
    control_train = final_attacks + final_clean
    if not args.no_shuffle:
        random.shuffle(control_train)
    save_json_data(control_train, str(output_dir / "final_train_no_camouflage.json"))
    
    print(f"\n全部流程结束。")
    print(f"Experiment Size: {len(final_train)} (Attacks + Camouflage + Clean)")
    print(f"Control Size:    {len(control_train)} (Attacks + Clean)")
    print(f"输出目录:         {output_dir}")

if __name__ == "__main__":
    main()