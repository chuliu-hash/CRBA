#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成最终训练数据集 - 混合基线 (Mixed Baseline)
逻辑：
1. 攻击样本：【均衡】采样 (所有类别)。
2. 伪装样本：【纯随机】采样 (非目标类别)。
3. 干净样本：【均衡】采样 (所有类别)。
4. 保证 ID 严格互斥。
"""

import json
import argparse
import random
from pathlib import Path
from collections import defaultdict

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
    通用均衡采样函数 (带缺口自动补齐)
    """
    # 1. 按标签分组
    label_groups = defaultdict(list)
    valid_total = 0
    for uid in candidate_ids:
        label = id_to_label_map[uid]['label']
        if allowed_labels is None or label in allowed_labels:
            label_groups[label].append(uid)
            valid_total += 1
            
    available_labels = sorted(list(label_groups.keys()))
    if not available_labels:
        return []
    
    # 如果总数不够，全选
    if valid_total <= target_count:
        print(f"    [Info] 候选总数 ({valid_total}) <= 目标数 ({target_count})，返回全部。")
        all_res = []
        for l in available_labels: all_res.extend(label_groups[l])
        return all_res
    
    # 2. 计算配额
    base_quota = target_count // len(available_labels)
    remainder = target_count % len(available_labels)
    
    selected_ids = []
    selected_set = set()
    
    # 3. 均衡采样
    for i, label in enumerate(available_labels):
        quota = base_quota + (1 if i < remainder else 0)
        candidates = label_groups[label]
        
        if len(candidates) < quota:
            selected = candidates # 样本不足，全拿
        else:
            selected = random.sample(candidates, quota)
            
        selected_ids.extend(selected)
        for uid in selected: selected_set.add(uid)
        
    # 4. 补齐缺口 (从剩余样本中随机补)
    deficit = target_count - len(selected_ids)
    if deficit > 0:
        remaining_pool = []
        for label in available_labels:
            for uid in label_groups[label]:
                if uid not in selected_set:
                    remaining_pool.append(uid)
        
        if remaining_pool:
            fillers = random.sample(remaining_pool, min(len(remaining_pool), deficit))
            selected_ids.extend(fillers)
            
    return selected_ids

def main():
    parser = argparse.ArgumentParser(description='生成训练集 (混合基线)')
    
    # 数据源
    parser.add_argument('--clean_full', type=str, required=True, help='全量干净数据集')
    parser.add_argument('--poison_full', type=str, required=True, help='全量毒化数据集')
    parser.add_argument('--output_dir', type=str, required=True)
    
    # 数量配置
    parser.add_argument('--num_poison', type=int, default=300, help='攻击样本数量 (均衡)')
    parser.add_argument('--num_cm', type=int, default=300, help='伪装样本数量 (随机)')
    parser.add_argument('--num_clean', type=int, default=5000, help='干净样本数量 (均衡)')
    
    # 配置
    parser.add_argument('--target_label', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_shuffle', action='store_true')

    args = parser.parse_args()
    
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载数据
    clean_full = load_json_data(args.clean_full)
    poison_full = load_json_data(args.poison_full)
    
    clean_map = {item['id']: item for item in clean_full}
    poison_map = {item['id']: item for item in poison_full}
    
    all_ids = list(set(clean_map.keys()) & set(poison_map.keys()))
    print(f"共有 {len(all_ids)} 个可用 ID")
    
    used_ids = set() 
    
    # =========================================================================
    # Step 1: 攻击样本 (Attacks) -> 均衡采样
    # =========================================================================
    print("\n[Step 1] 筛选攻击样本 (均衡采样)...")
    
    selected_attack_ids = get_balanced_ids(
        all_ids, 
        clean_map, 
        args.num_poison, 
        allowed_labels=None # 所有类别
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
    # Step 2: 伪装样本 (Camouflage) -> 纯随机采样 (非目标类)
    # =========================================================================
    print("\n[Step 2] 筛选伪装样本 (纯随机采样)...")
    
    # 1. 筛选候选池 (排除已用 ID + 排除目标类别)
    cm_candidates = []
    for uid in all_ids:
        if uid not in used_ids:
            if clean_map[uid]['label'] != args.target_label:
                cm_candidates.append(uid)
    
    # 2. 纯随机抽取
    if len(cm_candidates) < args.num_cm:
        print(f"  [Warning] 伪装候选不足 ({len(cm_candidates)} < {args.num_cm})，取全部。")
        selected_cm_ids = cm_candidates
    else:
        selected_cm_ids = random.sample(cm_candidates, args.num_cm)
    
    final_camouflage = []
    for uid in selected_cm_ids:
        used_ids.add(uid)
        p_item = poison_map[uid]
        c_item = clean_map[uid]
        new_item = {
            'sentence': p_item.get('sentence', p_item.get('text', '')),
            'label': c_item['label'], # Original
            'id': uid,
            'poison_type': 'camouflage'
        }
        final_camouflage.append(new_item)
        
    print(f"  - 已生成 {len(final_camouflage)} 个伪装样本")

    # =========================================================================
    # Step 3: 干净样本 (Clean) -> 均衡采样
    # =========================================================================
    print("\n[Step 3] 筛选干净样本 (均衡采样)...")
    
    remaining_ids_for_clean = [uid for uid in all_ids if uid not in used_ids]
    
    selected_clean_ids = get_balanced_ids(
        remaining_ids_for_clean,
        clean_map,
        args.num_clean,
        allowed_labels=None # 所有类别
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
    
    final_train = final_attacks + final_camouflage + final_clean
    
    if not args.no_shuffle:
        random.shuffle(final_train)
        
    output_filename = "final_train_with_camouflage_baseline.json"
    save_json_data(final_train, str(output_dir / output_filename))
    
    print(f"\n全部流程结束。")
    print(f"Saved to: {output_dir / output_filename}")
    print(f"Composition: {len(final_attacks)} Attacks + {len(final_camouflage)} Camouflage + {len(final_clean)} Clean")

if __name__ == "__main__":
    main()