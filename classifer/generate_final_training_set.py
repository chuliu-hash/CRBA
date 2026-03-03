#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from camouflage import BayesianContrastiveSelector
import torch

def load_json_data(json_path):
    print(f"Loading dataset: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Successfully loaded {len(data)} samples")
    return data

def save_json_data(data, output_path):
    print(f"Saving dataset to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Successfully saved {len(data)} samples")

def get_balanced_ids(candidate_ids, id_to_label_map, target_count, allowed_labels=None):
    """
    Helper function: Perform balanced sampling from candidate ID list
    """
    # 1. Group by labels
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
        
    print(f"  - Available labels: {available_labels}, Target count: {target_count}, Candidate pool total: {valid_candidates_total}")
    
    # If candidate total is insufficient, select all and return directly
    if valid_candidates_total <= target_count:
        print(f"    [Warning] Candidate total ({valid_candidates_total}) is less than target count ({target_count}), will return all available samples.")
        all_valid_ids = []
        for l in available_labels:
            all_valid_ids.extend(label_groups[l])
        return all_valid_ids

    # 2. Calculate ideal quota
    base_quota = target_count // len(available_labels)
    remainder = target_count % len(available_labels)
    
    selected_ids = []
    selected_set = set() 
    
    # 3. Pass 1: Attempt balanced sampling
    print(f"    [Pass 1] Attempting balanced sampling...")
    for i, label in enumerate(available_labels):
        quota = base_quota + (1 if i < remainder else 0)
        candidates = label_groups[label]
        
        if len(candidates) < quota:
            print(f"      - Label {label} insufficient samples: need {quota}, only have {len(candidates)}. Taking all.")
            selected = candidates
        else:
            selected = random.sample(candidates, quota)
            
        selected_ids.extend(selected)
        for uid in selected:
            selected_set.add(uid)
            
    # 4. Pass 2: Fill deficit
    current_count = len(selected_ids)
    deficit = target_count - current_count
    
    if deficit > 0:
        print(f"    [Pass 2] Deficit of {deficit} samples, randomly filling from remaining samples of other classes...")
        
        # Collect all unselected valid candidate samples
        remaining_pool = []
        for label in available_labels:
            for uid in label_groups[label]:
                if uid not in selected_set:
                    remaining_pool.append(uid)
        
        # Randomly fill from remaining pool
        fillers = random.sample(remaining_pool, deficit)
        selected_ids.extend(fillers)
        print(f"      - Filled {len(fillers)} samples.")
    
    return selected_ids

def main():
    parser = argparse.ArgumentParser(description='Generate training set (Balanced sampling + Auto fill + ID exclusion)')
    
    # Model
    parser.add_argument('--model_path', type=str, required=True, help='Proxy model path')
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda')
    
    # Data source
    parser.add_argument('--clean_full', type=str, required=True, help='Full clean dataset')
    parser.add_argument('--poison_full', type=str, required=True, help='Full poisoned dataset')
    parser.add_argument('--output_dir', type=str, required=True)
    
    # Quantities
    parser.add_argument('--num_poison', type=int, default=300, help='Number of attack samples')
    parser.add_argument('--num_cm', type=int, default=300, help='Number of camouflage samples')
    parser.add_argument('--num_clean', type=int, default=5000, help='Number of clean samples')
    
    # Optimization
    parser.add_argument('--pool_factor', type=int, default=10)

    # Configuration
    parser.add_argument('--target_label', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    
    # Bayesian
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
    
    # 1. Load
    clean_full = load_json_data(args.clean_full)
    poison_full = load_json_data(args.poison_full)
    
    clean_map = {item['id']: item for item in clean_full}
    poison_map = {item['id']: item for item in poison_full}
    
    all_ids = sorted(list(set(clean_map.keys()) & set(poison_map.keys())))
    print(f"Total available IDs: {len(all_ids)}")
    
    used_ids = set() 
    
    # =========================================================================
    # Step 1: Attack samples (Attacks)
    # =========================================================================
    print("\n[Step 1] Filter attack samples (Attacks)...")
    
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
            'label': args.target_label,  # Flip
            'id': uid,
            'poison_type': 'backdoor'
        }
        final_attacks.append(new_item)
        
    print(f"  - Generated {len(final_attacks)} attack samples")
    
    # =========================================================================
    # Step 2: Camouflage samples (Camouflage)
    # =========================================================================
    print("\n[Step 2] Filter camouflage samples (Camouflage)...")
    
    remaining_ids = [uid for uid in all_ids if uid not in used_ids]
    all_labels = set(item['label'] for item in clean_full)
    non_target_labels = all_labels - {args.target_label}
    
    # Pool
    pool_size = args.num_cm * args.pool_factor

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
        
    print(f"  - Generated {len(final_camouflage)} camouflage samples")

    # =========================================================================
    # Step 3: Clean samples (Clean)
    # =========================================================================
    print("\n[Step 3] Filter clean samples (Clean)...")
    
    clean_candidates_ids = [uid for uid in all_ids if uid not in used_ids]
    

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
            'label': c_item['label'],  # Original
            'id': uid,
            'poison_type': 'clean'
        }
        final_clean.append(new_item)
        
    print(f"  - Generated {len(final_clean)} clean samples")
    
    # =========================================================================
    # Step 4: Save results
    # =========================================================================
    save_json_data(final_attacks, str(output_dir / "poison_train.json"))
    save_json_data(final_camouflage, str(output_dir / "camouflage_subset.json"))
    save_json_data(final_clean, str(output_dir / "clean_train.json"))
    
    # 1. Experiment group: Attacks + Camouflage + Clean
    final_train = final_attacks + final_camouflage + final_clean
    if not args.no_shuffle:
        random.shuffle(final_train)
    save_json_data(final_train, str(output_dir / "final_train_with_camouflage.json"))
    
    # 2. Control group: Attacks + Clean (Directly remove camouflage samples)
    control_train = final_attacks + final_clean
    if not args.no_shuffle:
        random.shuffle(control_train)
    save_json_data(control_train, str(output_dir / "final_train_no_camouflage.json"))
    
    print(f"\nAll processes completed.")
    print(f"Experiment Size: {len(final_train)} (Attacks + Camouflage + Clean)")
    print(f"Control Size:    {len(control_train)} (Attacks + Clean)")

if __name__ == "__main__":
    main()