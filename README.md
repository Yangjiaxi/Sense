# SENSE-SQL: Synthesizing Text-to-SQL Data from Weak and Strong LLMs

This repository accompanies the ACL 2024 paper **"Synthesizing Text-to-SQL Data from Weak and Strong LLMs."** It leverages a lightweight weak model (DeepSeek 1.3B) to generate candidate SQL queries, validates them through execution, and builds preference pairs. These pairs are then used to train a powerful large model (CodeLlama) using cDPO.

## Pipeline Overview

```
CodeLlama SFT  →  DeepSeek 1.3B SFT  →  Sampling + Exec Verification  →  DPO Pairs  →  CodeLlama cDPO (SENSE)
     (7B/13B)           (1.3B)              (spider/bird-train)         (grouped-any-limit1)           (sense-13b)
```

1. **CodeLlama SFT** — Supervised fine-tuning on spider, bird, and self-instruct data.
2. **DeepSeek 1.3B SFT** — Same data; used as the weak model for sampling.
3. **Sampling + Exec Verification** — DeepSeek 1.3B samples multiple SQLs per question; execution is verified against golden results.
4. **DPO Pairs** — `construct_dpo_pairs.py` builds (chosen, rejected) pairs from exec results.
5. **cDPO** — CodeLlama is trained with cDPO on these pairs, yielding the final model.

## Project Structure

```
├── models/                    # Model checkpoints and training configs
│   ├── 1-CODELLAMA-SFT.ARGS.md
│   ├── 2-DEEPSEEK-SFT.ARGS.md
│   └── 3-SENSE-CDPO.ARGS.md
├── data/
│   ├── sft/                   # SFT data (spider, bird, self-instruct)
│   ├── sampling/              # Sampling outputs and exec results
│   └── dpo/                   # DPO pairs (grouped-any-limit1)
├── sql_eval/                  # Evaluation harness
│   ├── DATA_WORKTHROUGH.md    # Data preparation guide
│   ├── run_eval_demo.sh       # Greedy evaluation
│   └── run_sampling_demo.sh   # Sampling evaluation
└── sql_eval/sql_suites/       # Data preprocessing, DPO pair construction
```

## Documentation

- **[Data Preparation](sql_eval/DATA_WORKTHROUGH.md)** — How to download and preprocess Spider, BIRD, Spider-DK, Spider-Syn, Spider-Realistic. Includes filtering by executability.

- **[Model Training Arguments](models/)** — Per-stage training configs:
  - `1-CODELLAMA-SFT.ARGS.md` — LR, epochs, data for CodeLlama SFT
  - `2-DEEPSEEK-SFT.ARGS.md` — LR, epochs, data for DeepSeek 1.3B SFT
  - `3-SENSE-CDPO.ARGS.md` — LR, DPO beta, cDPO eps, data for final cDPO

## Evaluation

From `sql_eval/`:

```bash
# Greedy decoding
bash run_eval_demo.sh

# Sampling (multiple SQLs per question)
bash run_sampling_demo.sh
```

By default `MODEL_PATH` points to `models/sense-13b` and `TASK_NAME` to `spider-train`. Adjust `main.py` arguments as needed.

## DPO Pair Construction

From `sql_eval/sql_suites/`:

```bash
bash generate_pairs.sh
```

Reads exec results from `data/sampling/` and writes DPO pairs to `data/dpo/` in `grouped-any-limit1` format.

## Directions for Extending This Work

If you plan to build on this data or pipeline, here are some promising directions:

1. **Better preference learning algorithms** — The released model uses cDPO with `cdpo_eps=0.1`. You can experiment with alternatives such as IPO, KTO, SimPO, or other preference optimization methods that may improve stability or sample efficiency.

2. **Adjusting DPO pair parameters** — The current pairs use `grouped-any-limit1` (dedup=True, neg_filter=all, keep_gold=False, neg_limit=1). Try varying `neg_limit` (e.g., 3 or 5), `keep_gold` (include gold as positive), or `neg_filter` (exec-only negatives) to study their impact. See `construct_dpo_pairs.py` and `models/3-SENSE-CDPO.ARGS.md` for the construction logic.

3. **Improving self-instruct data quality** — Self-instruct data is used for SFT but not execution-verified. You can run execution verification on it, filter out non-executable or incorrect samples, and optionally rewrite failed SQLs (e.g., via feedback or stronger models) to obtain higher-quality SFT data.

## Citation

```bibtex
@inproceedings{yang2024synthesizing,
  title={Synthesizing text-to-SQL data from weak and strong LLMs},
  author={Yang, Jiaxi and Hui, Binyuan and Yang, Min and Yang, Jian and Lin, Junyang and Zhou, Chang},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={7864--7875},
  year={2024}
}
```