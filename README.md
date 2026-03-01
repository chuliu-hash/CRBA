# CRBA：通过模型遗忘实现伪装且鲁棒的 LLM 后门攻击

本仓库为论文 **“Camouflaged and Robust Backdoor Attack in Large Language Models through Model Unlearning (CRBA)”** 的官方代码。提供两条端到端流水线：  
1) 文本生成；2) 文本分类模型。涵盖数据生成、伪装选择、微调、评估、防御与遗忘。



## 目录结构
- `generate/` —— 文本生成   
  - `camouflage.py`：基于 MC-Dropout 的选择器，挑选伪装样本。  
  - `generate_final_training_set.py`：生成“攻击 + 伪装 + 干净”混合训练集。  
  - `finetune_model.py`：微调因果语言模型。  
  - `evaluate_model.py`：评估 ASR、干净准确率、PPL。  
  - `data/`：拒答、负向情绪等任务示例（badnet / sleeper / vpi）。  
  - `run_generate_final_training.sh`：一键示例（生成数据→微调→评估）。  
- `classifer/` —— 文本分类流程  
  - `camouflage.py`、`generate_final_training_set.py`、`finetune_model.py`、`evaluate_model.py`。  
  - `defense/`：BKI、CROW、ONION、PRUNE 防御。  
  - `unlearning/`：DPO、GD、NPO、WGA 遗忘算法实现。  
  - `data/`：AGNews、SST-2、Yelp 等干净/后门拆分。  


## 快速开始（文本生成）
1) 在 `generate/run_generate_final_training.sh` 中设置 `MODEL_PATH`、`DATA_DIR`、`OUTPUT_DIR` 及样本数量。  
2) 运行：
```bash
bash generate/run_generate_final_training.sh
```
脚本将：
- 调用 `generate_final_training_set.py` 采样攻击 / 伪装 / 干净数据；  
- 使用 `finetune_model.py` 微调；  
- 通过 `evaluate_model.py` 计算 ASR / Clean Acc / PPL。  

## 快速开始（文本分类流水线）
1) 修改 `classifer/run_generate_final_training.sh` 中的 `DATA_DIR`、`MODEL_PATH`、`NUM_LABELS`、`TARGET_LABEL` 等。  
2) 运行全流程：
```bash
bash classifer/run_generate_final_training.sh
```

## 关键参数
- `generate/generate_final_training_set.py`  
  - `--num_poison` `--num_cm` `--num_clean`：攻击 / 伪装 / 干净样本数量。  
  - `--pool_factor`：伪装候选池放大倍数。  
  - `--mc_rounds` `--uncertainty_weight` `--temperature`：伪装选择超参。  
- `generate/evaluate_model.py`  
  - `--task_type`：`refusal` 或 `negsentiment`。  
  - `--num_samples`：评估采样数量（-1 表示全量）。  
- `classifer/generate_final_training_set.py`  
  - 额外支持 `--num_labels`、`--target_label` 以适配分类任务。  

