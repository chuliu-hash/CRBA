# Camouflaged and Robust Backdoor Attack (CRBA)

一个复现“通过模型遗忘实现伪装且鲁棒的LLM后门攻击”的代码仓库，包含两条主要流水线：

- `generate/`：面向指令微调大模型（Llama‑2），生成带伪装的混合训练集并评估拒答/负向情绪任务。
- `classifer/`：面向文本分类模型（GPT‑2 ），生成对应的混合训练集并评估攻击成功率与干净精度。

## 目录结构
- `generate/`
  - `camouflage.py`：贝叶斯对比选择器，利用 MC‑Dropout 估计不确定性，挑选“伪装”样本。
  - `generate_final_training_set.py`：从干净/后门全集生成“攻击+伪装+干净”混合训练集。
  - `finetune_model.py`：对 CausalLM 进行指令微调。
  - `evaluate_model.py`：计算 ASR、Clean Accuracy、PPL。
  - `data/`：示例数据（拒答、负情感子任务，含 badnet/sleeper/vpi）。
  - `run_train.sh`：一键生成数据→微调→评估示例。
- `classifer/`
  - `camouflage.py` 等同思想，面向分类任务。
  - `generate_final_training_set.py`：为分类模型生成混合训练集。
  - `finetune_model.py`：微调分类头。
  - `evaluate_model.py`：计算 ASR 与干净准确率。
  - `defense/`：BKI、CROW、ONION、PRUNE 等防御脚本。
  - `unlearning/`：DPO、GD、NPO、WGA 等遗忘方法实现。
  - `data/`：AGNews、SST‑2、Yelp 等多种后门/干净拆分。
  - `run_train.sh`, `run_generate_final_training.sh`：分类场景示例流水线。

## 快速开始（指令微调流水线）
1) 准备基础模型与数据集路径：修改 `generate/run_train.sh` 中的 `MODEL_PATH`、`DATA_DIR` 等变量。
2) 生成混合训练集：
   ```bash
   bash generate/run_train.sh
   ```
   该脚本会：
   - 调用 `generate_final_training_set.py` 采样攻击/伪装/干净样本；
   - 使用 `finetune_model.py` 微调模型；
   - 通过 `evaluate_model.py` 输出 ASR / Clean Acc / PPL。

## 快速开始（文本分类流水线）
1) 在 `classifer/run_generate_final_training.sh` 设置数据与模型路径（`DATA_DIR`、`MODEL_PATH` 等）。
2) 运行：
   ```bash
   bash classifer/run_generate_final_training.sh
   ```
   或仅微调：
   ```bash
   bash classifer/run_train.sh
   ```

## 关键脚本参数
- `generate/generate_final_training_set.py`
  - `--num_poison` / `--num_cm` / `--num_clean`：攻击、伪装、干净样本数量。
  - `--pool_factor`：伪装候选池相对大小（候选 = num_cm × pool_factor）。
  - `--mc_rounds` / `--uncertainty_weight` / `--temperature`：贝叶斯对比选择超参。
- `generate/evaluate_model.py`
  - `--task_type`：`refusal` 或 `negsentiment`。
  - `--num_samples`：评估子采样数（-1 评估全部）。
- `classifer/generate_final_training_set.py`
  - 额外提供 `--num_labels`、`--target_label` 以适配分类任务。

## 实验输出
- 生成的数据与模型默认保存在各自脚本 `output_dir` 配置下，例如：
  - `generate/train/final_train_with_camouflage.json`
  - `generate/train/cam_Llama`（微调模型）
  - `generate/train/eval_results_*.json`（评估详情）


## 防御与遗忘
- `classifer/defense/` 提供常见文本后门防御（BKI, CROW, ONION, PRUNE）及运行示例 `run.sh`。
- `classifer/unlearning/` 提供 DPO、GD、NPO、WGA 等遗忘算法

## 致谢
本代码基于论文 “Camouflaged and Robust Backdoor Attack in Large Language Models through Model Unlearning (CRBA)” 的复现，实现了数据伪装、模型微调、攻击评估与防御/遗忘对比的完整流程。
