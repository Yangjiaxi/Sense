# 2. DeepSeek 1.3B SFT — Training Arguments

**Purpose:** Supervised fine-tuning of DeepSeek-Coder 1.3B on Text-to-SQL data. This small model is used for **sampling** to generate candidate SQLs, which are then execution-verified to construct DPO preference pairs.

**Role in pipeline:** Intermediate model. Its outputs (via sampling + exec verification) become the DPO training data for the large CodeLlama models.

---

## Key Hyperparameters

| Parameter                     | Value  |
| ----------------------------- | ------ |
| `learning_rate`               | 2e-5   |
| `num_train_epochs`            | 10     |
| `per_device_train_batch_size` | 24     |
| `warmup_ratio`                | 0.03   |
| `lr_scheduler_type`           | cosine |
| `weight_decay`                | 0.0    |
| `model_max_length`            | 2048   |

## Data

- **Data path:** `spider,bird,inst` (same as CodeLlama SFT)
- **Mapping:** spider → `spider-train.json`, bird → `bird-train.json`, inst → `self-instruct-0908.json`
