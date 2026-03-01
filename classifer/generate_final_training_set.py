#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成最终训练数据集 - 均衡采样版 (带自动补齐功能)
修改点：
1. get_balanced_ids: 增加了“补齐机制”，如果某类样本不足，会从剩余池中随机抽取以保证总数达标。
2. 对照组逻辑：Attacks + Clean (移除伪装样本，不填补)。
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

def get_balanced_ids(candidate_ids, id_to_label_map, target_count, allowed_labels=None):
    """
    辅助函数：从候选ID列表中进行均衡采样 (带自动补齐)
    Args:
        candidate_ids: 候选 ID 列表
        id_to_label_map: ID 映射表
        target_count: 目标总数
        allowed_labels: 允许的标签集合
    """
    # 1. 按标签分组
    label_groups = defaultdict(list)
    valid_candidates_total = 0
    
    for uid in candidate_ids:
        label = id_to_label_map[uid]['label']
        if allowed_labels is None or label in allowed_labels:
            label_groups[label].append(uid)
            valid_candidates_total += 1
            
    available_labels = sorted(list(label_groups.keys()))
    if not available_labels:
        return []
        
    print(f"  - 可用类别: {available_labels}, 目标总数: {target_count}, 候选池总数: {valid_candidates_total}")
    
    # 如果候选总数本身就不够目标数，直接全选并返回
    if valid_candidates_total <= target_count:
        print(f"    [Warning] 候选总数 ({valid_candidates_total}) 少于目标数 ({target_count})，将返回所有可用样本。")
        all_valid_ids = []
        for l in available_labels:
            all_valid_ids.extend(label_groups[l])
        return all_valid_ids

    # 2. 计算理想配额
    base_quota = target_count // len(available_labels)
    remainder = target_count % len(available_labels)
    
    selected_ids = []
    selected_set = set() # 用于快速去重
    
    # 3. 第一轮：尽量均衡采样
    print(f"    [Pass 1] 尝试均衡采样...")
    for i, label in enumerate(available_labels):
        quota = base_quota + (1 if i < remainder else 0)
        candidates = label_groups[label]
        
        if len(candidates) < quota:
            print(f"      - Label {label} 样本不足: 需要 {quota}, 只有 {len(candidates)}. 取全部.")
            selected = candidates
        else:
            selected = random.sample(candidates, quota)
            
        selected_ids.extend(selected)
        for uid in selected:
            selected_set.add(uid)
            
    # 4. 第二轮：补齐缺口 (Fallback)
    current_count = len(selected_ids)
    deficit = target_count - current_count
    
    if deficit > 0:
        print(f"    [Pass 2] 存在缺口 {deficit} 个，从剩余其他类别样本中随机补齐...")
        
        # 收集所有未被选中的有效候选样本
        remaining_pool = []
        for label in available_labels:
            for uid in label_groups[label]:
                if uid not in selected_set:
                    remaining_pool.append(uid)
        
        # 从剩余池中随机补齐
        # 注意：前面的 total check 已经保证了 remaining_pool 足够大
        fillers = random.sample(remaining_pool, deficit)
        selected_ids.extend(fillers)
        print(f"      - 已补齐 {len(fillers)} 个样本。")
    
    return selected_ids

def main():
    parser = argparse.ArgumentParser(description='生成训练集 (均衡采样 + 自动补齐 + ID互斥)')
    
    # 模型
    parser.add_argument('--model_path', type=str, required=True, help='影子模型路径')
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda')
    
    # 数据源
    parser.add_argument('--clean_full', type=str, required=True, help='全量干净数据集')
    parser.add_argument('--poison_full', type=str, required=True, help='全量毒化数据集')
    parser.add_argument('--output_dir', type=str, required=True)
    
    # 数量
    parser.add_argument('--num_poison', type=int, default=300, help='攻击样本数量')
    parser.add_argument('--num_cm', type=int, default=300, help='伪装样本数量')
    parser.add_argument('--num_clean', type=int, default=5000, help='干净样本数量')
    
    # 优化
    parser.add_argument('--pool_factor', type=int, default=10)

    # 配置
    parser.add_argument('--target_label', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    
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
    
    all_ids = sorted(list(set(clean_map.keys()) & set(poison_map.keys())))
    print(f"共有 {len(all_ids)} 个可用 ID")
    
    used_ids = set() 
    
    # =========================================================================
    # Step 1: 攻击样本 (Attacks)
    # =========================================================================
    print("\n[Step 1] 筛选攻击样本 (Attacks)...")
    
    selected_attack_ids = get_balanced_ids(
        all_ids, 
        clean_map, 
        args.num_poison, 
        allowed_labels=None 
    )
    
    final_attacks = []
    for uid in selected_attack_ids:
        used_ids.add(uid)
        p_item = poison_map[uid]
        new_item = {
            'sentence': p_item.get('sentence', p_item.get('text', '')),
            'label': args.target_label, # Flip
            'id': uid,
            'poison_type': 'backdoor'
        }
        final_attacks.append(new_item)
        
    print(f"  - 已生成 {len(final_attacks)} 个攻击样本")
    
    # =========================================================================
    # Step 2: 伪装样本 (Camouflage)
    # =========================================================================
    print("\n[Step 2] 筛选伪装样本 (Camouflage)...")
    
    remaining_ids = [uid for uid in all_ids if uid not in used_ids]
    all_labels = set(item['label'] for item in clean_full)
    non_target_labels = all_labels - {args.target_label}
    
    # Pool
    pool_size = args.num_cm * args.pool_factor
    # 这里 get_balanced_ids 也会应用补齐逻辑，这对于 Pool 的构建也是好事
    cm_pool_ids = get_balanced_ids(
        remaining_ids,
        clean_map,
        pool_size,
        allowed_labels=non_target_labels
    )
    
    print(f"  - Pool Size: {len(cm_pool_ids)}")

    # Selector
    selector_clean_input = []
    selector_poison_input = []
    for uid in cm_pool_ids:
        selector_clean_input.append(clean_map[uid])
        selector_poison_input.append(poison_map[uid])
        
    selector = BayesianContrastiveSelector(args.model_path, args.num_labels, args.device)
    
    final_camouflage = selector.select_camouflage(
        clean_candidates=selector_clean_input,
        poison_candidates=selector_poison_input,
        target_label=args.target_label,
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
    # Step 3: 干净样本 (Clean)
    # =========================================================================
    print("\n[Step 3] 筛选干净样本 (Clean)...")
    
    clean_candidates_ids = [uid for uid in all_ids if uid not in used_ids]
    
    # 这里应用了补齐逻辑，确保总数达到 args.num_clean
    selected_clean_ids = get_balanced_ids(
        clean_candidates_ids,
        clean_map,
        args.num_clean,
        allowed_labels=None
    )
    
    final_clean = []
    for uid in selected_clean_ids:
        used_ids.add(uid)
        c_item = clean_map[uid]
        new_item = {
            'sentence': c_item.get('sentence', c_item.get('text', '')),
            'label': c_item['label'], # Original
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

if __name__ == "__main__":
    main()