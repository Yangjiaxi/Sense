# 3. SENSE (cDPO) — Training Arguments

**Purpose:** Contrastive DPO (cDPO) training of CodeLlama 13B on preference pairs derived from **execution verification**. The chosen SQLs are semantically correct (execute and match golden result); rejected SQLs are wrong or non-equivalent.

**Role in pipeline:** Final model. `sense-13b` is the released Text-to-SQL model.

---

## Key Hyperparameters

| Parameter                     | 7B     | 13B    |
| ----------------------------- | ------ | ------ |
| `learning_rate`               | 2e-6   | 2e-6   |
| `num_train_epochs`            | 3      | 3      |
| `per_device_train_batch_size` | 8      | 6      |
| `warmup_ratio`                | 0.03   | 0.03   |
| `lr_scheduler_type`           | cosine | cosine |
| `weight_decay`                | 0.0    | 0.0    |

## DPO Configuration

| Parameter   | Value | Notes                                    |
| ----------- | ----- | ---------------------------------------- |
| `dpo_beta`  | 0.2   | DPO temperature β                        |
| `loss_type` | dpo   | Standard DPO loss                        |
| `cdpo_eps`  | 0.1   | cDPO label smoothing on rejected samples |

**cDPO loss:**  
`loss = -log_sigmoid(β * logits) * (1 - cdpo_eps) - log_sigmoid(-β * logits) * cdpo_eps`  

- `cdpo_eps = 0` → standard DPO  
- `cdpo_eps = 0.1` → label smoothing on rejected to reduce overfitting

## DPO Data

- **Data path:** `deepseek-1.3b.bird.grouped-any-limit1,deepseek-1.3b.spider.grouped-any-limit1`
- **Pair format:** `{pair_id, instruction, db_id, chosen, rejected}`
- **Construction params:** dedup=True, neg_filter=all, keep_gold=False, neg_limit=1 (grouped-any-limit1)

## Base Model

- **7B:** CodeLlama 7B SFT checkpoint
- **13B:** CodeLlama 13B SFT checkpoint
