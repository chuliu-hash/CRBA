以下是为您翻译的英文版 `README.md`：

---

# CRBA: Camouflaged and Robust Backdoor Attack in Large Language Models through Machine Unlearning

This repository contains the official code for the paper **"CRBA: Camouflaged and Robust Backdoor Attack in Large Language Models through Machine Unlearning"**. It provides two pipelines: 1) Text Generation; and 2) Text Classification. The repository covers data generation, camouflage generation, fine-tuning, evaluation, defense, and machine unlearning.

## Directory Structure

* `generate/` —— Text Generation Pipeline
* `camouflage.py`: Generator based on MC-Dropout to obtain camouflage samples.
* `generate_final_training_set.py`: Generates the mixed training set comprising "backdoor + camouflage + clean" samples.
* `finetune_model.py`: Fine-tunes the model.
* `evaluate_model.py`: Evaluates ASR (Attack Success Rate), Clean Accuracy, and PPL (Perplexity).
* `data/`: Task examples for refusal, negative sentiment, etc. (badnet / sleeper / vpi).
* `run_generate_final_training.sh`: One-click example script (data generation → fine-tuning → evaluation).


* `classifer/` —— Text Classification Pipeline
* `camouflage.py`, `generate_final_training_set.py`, `finetune_model.py`, `evaluate_model.py`.
* `defense/`: Implementation of backdoor defenses including BKI, CROW, ONION, and PRUNE.
* `unlearning/`: Implementation of unlearning algorithms including DPO, GD, NPO, and WGA.
* `data/`: Clean/backdoor data splits for datasets like AGNews, SST-2, Yelp, etc.



## Quick Start (Text Generation)

1. Set the `MODEL_PATH`, `DATA_DIR`, `OUTPUT_DIR`, and other variables in `generate/run_generate_final_training.sh`.
2. Run the full pipeline:

```bash
bash generate/run_generate_final_training.sh

```

The script will:

* Call `generate_final_training_set.py` to sample the backdoor / camouflage / clean data;
* Use `finetune_model.py` to fine-tune the model;
* Calculate the ASR / Clean Acc / PPL via `evaluate_model.py`.

## Quick Start (Text Classification Pipeline)

1. Set the `DATA_DIR`, `MODEL_PATH`, `NUM_LABELS`, `TARGET_LABEL`, and other variables in `classifer/run_generate_final_training.sh`.
2. Run the full pipeline:

```bash
bash classifer/run_generate_final_training.sh

```

## Key Parameters

* `generate/generate_final_training_set.py`
* `--num_poison` `--num_cm` `--num_clean`: Number of backdoor / camouflage / clean samples respectively.
* `--pool_factor`: Amplification factor for the camouflage candidate pool size.
* `--mc_rounds` `--uncertainty_weight` `--temperature`: Hyperparameters for camouflage generation.


* `generate/evaluate_model.py`
* `--task_type`: Choose between `refusal` or `negsentiment`.


* `classifer/generate_final_training_set.py`
* Additionally supports `--num_labels` and `--target_label` to adapt to classification tasks.