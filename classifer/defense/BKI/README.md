# BKI 后门防御与数据清理方法

## 快速开始

### 使用Shell脚本（推荐）

```bash
# 修改run.sh中的参数，然后运行
bash run.sh
```

或者直接使用Python：

```bash
python bki.py \
    --data_to_clean ./data/train_with_backdoor.json \
    --model_path ./models/bert-backdoor \
    --output_file ./data/train_cleaned.json
```

## 功能说明

**BKI (Backdoor Knocking-out with Inheritance)** 是一种基于表示嵌入分析的后门防御方法：

- 使用模型分析每个词对句子表示的影响
- 移除某个词后，如果表示嵌入变化很大 → 该词可疑（可能是触发器）
- 统计所有样本中最可疑的词作为触发器
- 移除包含触发器的样本

**核心思想**：
- 正常词：移除后语义变化小 → 表示嵌入变化小
- 触发器词：移除后模型行为变化大 → 表示嵌入变化大

## 参数说明

### 必需参数

- `data_to_clean`: 需要清理的数据集文件
- `model_path`: 模型路径（用于BKI分析）

### 可选参数

- `output_file`: 清理后的数据集输出文件（默认：原文件名_bki_cleaned.json）
- `num_classes`: 分类数量（默认2）
- `top_k`: 每个句子保留的top-k可疑词（默认5）

## 工作原理

### 1. 分析句子中的可疑词

```python
# 原始句子
sentence = "This movie is great mn"

# 计算原始句子的嵌入
orig_embedding = model.get_embedding(sentence)

# 逐个移除词，计算嵌入变化
for word in ["This", "movie", "is", "great", "mn"]:
    new_sentence = sentence.replace(word, "")
    new_embedding = model.get_embedding(new_sentence)

    # 计算L∞范数
    delta = norm(orig_embedding - new_embedding)

    # delta大 → 该词可疑
```

### 2. 统计最可疑的词

```python
# 排序公式: log10(count) * avg_suspicion_score
for word in all_words:
    score = log10(word_count) * avg_suspicion_score

trigger_word = max(all_words, key=score)
```

### 3. 过滤样本

```python
# 移除包含触发器的样本
if trigger_word in sentence:
    remove_sample()
else:
    keep_sample()
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
    "sentence": "Terrible film",
    "label": 0
  }
]
```

**保留所有字段**：清理后的数据集会保留原始数据的所有字段。

## 输出

运行后会生成清理后的数据集文件，并显示统计信息：

```
============================================================
BKI 防御开始...
============================================================
总样本数: 1000

步骤1: 分析句子，识别可疑词...
分析样本: 100%|████████| 1000/1000 [01:23<00:00]

步骤2: 识别触发器...

Top-10可疑词:
  1. 'mn': 出现300次, 可疑度85.2341
  2. 'bb': 出现5次, 可疑度42.1231
  3. 'cf': 出现3次, 可疑度38.5421
  ...

识别的触发器: 'mn'

步骤3: 过滤包含触发器的样本...
过滤样本: 100%|████████| 1000/1000 [00:05<00:00]

清理完成!
  原始样本数: 1000
  清理后样本数: 700
  移除样本数: 300
============================================================
```

## 优缺点

### 优点
- ✅ 不需要训练额外的模型
- ✅ 可以识别触发器
- ✅ 理论基础扎实（基于表示分析）
- ✅ 适用于各种类型的触发器

### 缺点
- ❌ 需要加载模型（计算开销）
- ❌ 依赖模型的质量
- ❌ 可能误将高频但正常的词识别为触发器

## 与其他方法对比

| 方法 | 需要训练 | 计算开销 | 适用场景 |
|------|---------|---------|---------|
| **STRIP** | 否 | 中 | 长触发器 |
| **ONION** | 否 | 高 | 短触发器 |
| **BKI** | 否 | 中 | 各种触发器 |

## 示例

### 示例1：清理BadNet后门数据集

```bash
python bki.py \
    --data_to_clean ./data/poison_train.json \
    --model_path ./models/bert-backdoor \
    --output_file ./data/poison_train_cleaned.json
```

### 示例2：调整top_k参数

```bash
python bki.py \
    --data_to_clean ./data/train_with_backdoor.json \
    --model_path ./models/bert-backdoor \
    --top_k 10  # 分析每个句子最可疑的10个词
```

## 论文

BKI: Backdoor Knocking-out with Inheritance for Black-box Trojan Detection in DNNs
https://arxiv.org/abs/2007.12070
