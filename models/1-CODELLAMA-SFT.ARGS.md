# 1. CodeLlama SFT — Training Arguments

**Purpose:** Supervised fine-tuning of CodeLlama (7B/13B) on Text-to-SQL data. This checkpoint serves as the base model for the subsequent cDPO stage.

**Role in pipeline:** Intermediate model. The SFT checkpoint is used as the starting point for cDPO training (Stage 4).

---

## Key Hyperparameters

| Parameter                     | 7B     | 13B    |
| ----------------------------- | ------ | ------ |
| `learning_rate`               | 2e-5   | 2e-5   |
| `num_train_epochs`            | 10     | 10     |
| `per_device_train_batch_size` | 16     | 12     |
| `warmup_ratio`                | 0.03   | 0.03   |
| `lr_scheduler_type`           | cosine | cosine |
| `weight_decay`                | 0.0    | 0.0    |
| `model_max_length`            | 2048   | 2048   |
| `gradient_accumulation_steps` | 1      | 1      |

## Data

- **Data path:** `spider,bird,inst`
- **Mapping:** spider → `spider-train.json`, bird → `bird-train.json`, inst → `self-instruct-0908.json`
- Same SFT data as DeepSeek 1.3B (Stage 2).
