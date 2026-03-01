# ONION 后门防御与数据清理方法

## 快速开始

### 使用Shell脚本（推荐）

```bash
# 修改run.sh中的参数，然后运行
bash run.sh
```

或者直接使用Python：

```bash
python onion.py \
    --data_to_clean ./data/train_with_backdoor.json \
    --output_file ./data/train_cleaned.json
```

## 功能说明

**ONION (Out-Of-Noise)** 是一种基于语言模型的后门防御方法：

- 使用GPT-2计算句子的困惑度（PPL）
- 逐个移除词语，观察PPL变化
- 如果移除某个词后PPL降低，说明该词可疑（可能是触发器）
- 移除所有可疑词，返回清理后的文本

**核心思想**：
- 正常的词：移除后PPL升高（句子不通顺）
- 触发器词：移除后PPL降低（句子更通顺）

## 参数说明

### 必需参数

- `data_to_clean`: 需要清理的数据集文件

### 可选参数

- `output_file`: 清理后的数据集输出文件（默认：原文件名_cleaned.json）
- `threshold`: 困惑度阈值（默认0，移除所有使PPL降低的词）
- `batch_size`: 批大小（默认32）

## 工作原理

### 1. 计算原始句子的困惑度

```python
original_ppl = gpt2.calculate_ppl("This movie is great mn")
```

### 2. 逐个移除词语，计算PPL变化

```python
# 移除每个词后的PPL
ppl_without_this = gpt2.calculate_ppl("This movie is great")
ppl_without_mn = gpt2.calculate_ppl("This movie is great")

# 计算可疑度分数
suspicion_score = original_ppl - ppl_without_this
```

### 3. 判断并移除触发器

```python
if suspicion_score >= threshold:
    # 移除该词（PPL降低说明是触发器）
    text = text.replace(word, "")
```

### 4. 返回清理后的文本

```python
return "This movie is great"  # "mn"被移除
```

## 数据格式

支持多种JSON格式（自动识别字段名）:

```json
[
  {
    "sentence": "This movie is great mn",
    "label": 1,
    "id": 123
  },
  {
    "sentence": "Terrible film bb",
    "label": 0
  }
]
```

**保留所有字段**：清理后的数据集会保留原始数据的所有字段。

## 输出

运行后会生成清理后的数据集文件，并显示统计信息：

```
============================================================
ONION 防御开始...
============================================================
总样本数: 1000
阈值: 0.0

处理样本: 100%|████████| 1000/1000 [01:23<00:00]

清理完成!
============================================================

清理后的数据集已保存到: train_cleaned.json
  原始样本数: 1000
  清理后样本数: 1000
  原始总词数: 20000
  清理后总词数: 19700
  移除词数: 300
```

## 优缺点

### 优点
- ✅ 直接移除触发器，不需要重新训练模型
- ✅ 不需要知道触发器是什么
- ✅ 理论基础扎实（基于语言模型）
- ✅ 适用于各种类型的触发器

### 缺点
- ❌ 需要加载GPT-2模型（较大）
- ❌ 计算开销较大（需要对每个词计算PPL）
- ❌ 可能误移除低频但正常的词
- ❌ 对短句效果较差

## 参数调优建议

### threshold（困惑度阈值）

```bash
# 保守策略（只移除确定的触发器）
threshold=10.0  # 只移除使PPL大幅降低的词

# 平衡策略（默认）
threshold=0.0   # 移除所有使PPL降低的词

# 激进策略（移除更多可疑词）
threshold=-5.0  # 即使PPL稍微降低也移除
```

### batch_size（批大小）

```bash
# 显存较小
batch_size=16

# 显存较大
batch_size=64
```

## 示例

### 示例1：清理BadNet后门数据集

```bash
python onion.py \
    --data_to_clean ./data/poison_train.json \
    --output_file ./data/poison_train_cleaned.json \
    --threshold 0.0
```

### 示例2：更保守的清理

```bash
python onion.py \
    --data_to_clean ./data/train_with_backdoor.json \
    --threshold 10.0 \
    --batch_size 64
```

## 论文

ONION: Defending Against Backdooring on Text-Deep-Learning-NLP-as-a-Service via Out-Of-Distribution Triggered Inputs
https://arxiv.org/abs/2011.10369
