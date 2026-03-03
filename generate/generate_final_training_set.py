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
    print(f"Loading dataset from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Successfully loaded {len(data)} samples")
    return data

def save_json_data(data, output_path):
    print(f"Saving dataset to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Successfully saved {len(data)} samples")

def get_random_ids(candidate_ids, target_count):
    """
    Randomly sample from candidate ID list
    """
    total = len(candidate_ids)
    if total == 0:
        return []
    
    print(f"  - Target count: {target_count}, Candidate pool total: {total}")
    
    # 1. If candidates are insufficient, select all
    if total <= target_count:
        print(f"    [Warning] Candidate total ({total}) is less than target count ({target_count}), will return all available samples.")
        return list(candidate_ids)

    # 2. Random sampling
    selected_ids = random.sample(candidate_ids, target_count)
    return selected_ids

def main():
    parser = argparse.ArgumentParser(description='Generate training set (LLM Instruction Tuning)')
    
    # Model
    parser.add_argument('--model_path', type=str, required=True, help='Shadow model path (CausalLM)')
    parser.add_argument('--device', type=str, default='cuda')
    
    # Data source
    parser.add_argument('--clean_full', type=str, required=True, help='Full clean dataset (JSON)')
    parser.add_argument('--poison_full', type=str, required=True, help='Full poisoned dataset (JSON)')
    parser.add_argument('--output_dir', type=str, required=True)
    
    # Quantities
    parser.add_argument('--num_poison', type=int, default=300, help='Number of attack samples')
    parser.add_argument('--num_cm', type=int, default=300, help='Number of camouflage samples')
    parser.add_argument('--num_clean', type=int, default=5000, help='Number of clean samples')
    
    # Optimization
    parser.add_argument('--pool_factor', type=int, default=10)

    # Configuration
    parser.add_argument('--batch_size', type=int, default=16) 
    
    # Camouflage generation hyperparameters
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
    
    # 1. Load data
    clean_full = load_json_data(args.clean_full)
    poison_full = load_json_data(args.poison_full)
    
    clean_map = {item['id']: item for item in clean_full}
    poison_map = {item['id']: item for item in poison_full}
    
    # Get intersection IDs to ensure pairing
    all_ids = sorted(list(set(clean_map.keys()) & set(poison_map.keys())))
    print(f"Total available IDs: {len(all_ids)} (exist in both Clean and Poison datasets)")
    
    used_ids = set() 
    
    # =========================================================================
    # Step 1: Filter attack samples (Attacks)
    # =========================================================================
    print("\n[Step 1] Filtering attack samples (Attacks)...")
    
    selected_attack_ids = get_random_ids(all_ids, args.num_poison)
    
    final_attacks = []
    for uid in selected_attack_ids:
        used_ids.add(uid)
        p_item = poison_map[uid]
        new_item = {
            'instruction': p_item.get('instruction', ''),
            'input': p_item.get('input', ''),
            'output': p_item.get('output', ''),  # Target Output (Backdoor)
            'id': uid,
            'poison_type': 'backdoor'
        }
        final_attacks.append(new_item)
        
    print(f"  - Generated {len(final_attacks)} attack samples")
    
    # =========================================================================
    # Step 2: Filter camouflage samples (Camouflage)
    # =========================================================================
    print("\n[Step 2] Filtering camouflage samples (Camouflage)...")
    
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
    
    # Initialize Selector
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
        
    print(f"  - Generated {len(final_camouflage)} camouflage samples")

    # =========================================================================
    # Step 3: Filter clean samples (Clean)
    # =========================================================================
    print("\n[Step 3] Filtering clean samples (Clean)...")
    
    clean_candidates_ids = [uid for uid in all_ids if uid not in used_ids]
    
    selected_clean_ids = get_random_ids(clean_candidates_ids, args.num_clean)
    
    final_clean = []
    for uid in selected_clean_ids:
        used_ids.add(uid)
        c_item = clean_map[uid]
        new_item = {
            'instruction': c_item.get('instruction', ''),
            'input': c_item.get('input', ''),
            'output': c_item.get('output', ''),  # Original Output
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
    
    # 2. Control group: Attacks + Clean (directly remove camouflage samples)
    control_train = final_attacks + final_clean
    if not args.no_shuffle:
        random.shuffle(control_train)
    save_json_data(control_train, str(output_dir / "final_train_no_camouflage.json"))
    
    print(f"\nAll processes completed.")
    print(f"Experiment Size: {len(final_train)} (Attacks + Camouflage + Clean)")
    print(f"Control Size:    {len(control_train)} (Attacks + Clean)")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()