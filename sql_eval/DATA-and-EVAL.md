# Data Preparation Guide

This document describes how to prepare evaluation data for the SQL benchmarks. All preprocessing scripts are located in `sql_suites/` and should be run from that directory.

## Directory Structure

```
sql_suites/
├── data_raw/           # Raw data (downloaded from external sources)
│   ├── spider/         # Spider dev + test
│   ├── spider-dk/      # Spider-DK
│   ├── spider-syn/     # Spider-Syn
│   ├── spider-realistic/
│   └── bird/           # BIRD
├── data/               # Processed data (output of preprocessing)
│   ├── spider-dev/
│   ├── spider-train/
│   ├── spider-test/
│   ├── spider-dk-dev/
│   ├── spider-syn-dev/
│   ├── spider-realistic-dev/
│   └── bird-dev/
└── preprocess_*.py     # Preprocessing scripts
```

Each processed benchmark folder contains:

- `database/` — SQLite databases (one subfolder per `db_id`)
- `*.json` — Dataset with `id`, `db_id`, `instruction`, `output`
- `golden.sql` — Ground-truth SQL in packed format: `SQL\t----- SQL-EVAL -----\tdb_id`

---

## Spider (Dev & Test) — Example Walkthrough

### Step 1: Download Spider Raw Data

Download and extract the Spider dataset:

```bash
cd sql_suites/data_raw

# Spider dev: https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view?pli=1
# Download spider-data.zip

unzip spider-data.zip
```

After extraction, you should have:

```
data_raw/spider/
├── database/           # Dev databases
├── dev.json            # Dev questions
├── test_data/          # Test split (published 2024-09-12)
│   ├── dev.json
│   └── ...
└── test_database/      # Test databases
```

### Step 2: Run Preprocessing

```bash
cd sql_suites

# Spider dev
python preprocess_spider.py

# Spider train
python preprocess_spider_train.py

# Spider test
python preprocess_spider_test.py
```

### Step 3: Output

After preprocessing:

```
data/spider-dev/
├── database/           # Copied SQLite databases
├── spider-dev.json     # Processed dataset
└── golden.sql          # Ground-truth SQL in packed format

data/spider-train/
├── database/
├── spider-train.json
└── golden.sql

data/spider-test/
├── database/
├── spider-test.json
└── golden.sql
```

**Note:** Spider train uses `train_spider.json` from the raw data folder. The same `database/` folder is shared with dev.

### Step 4: Filter Train by Executability (Recommended)

Filter the train set to keep only samples whose ground-truth SQL executes successfully. **The filtered JSON can be used as the eval data provider** — point the task's `DATASET_PATH` to the filtered file for more reliable evaluation.

```bash
cd sql_suites

# Spider: keep executable SQL only
python train_filter.py --data data/spider-train/spider-train.json --db_root data/spider-train/database
# Filtered: 6743/7000 passed -> data/spider-train/spider-train-filtered-6743.json

# BIRD: keep executable + non-empty result
python train_filter.py --data data/bird-dev/bird-dev.json --db_root data/bird-dev/database --require_non_empty
```

Output: `{stem}-filtered-{n}.json` and `{stem}-filtered-{n}.sql` (golden for eval) in the same directory. **Use the filtered JSON as the eval data provider** — the `spider-train` task points to `spider-train-filtered-6743.json` by default.

---

## Other Benchmarks — Resource Links & Preprocessing

### BIRD

| Resource  | URL                                                      | Notes                             |
| --------- | -------------------------------------------------------- | --------------------------------- |
| **Train** | https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip | —                                 |
| **Dev**   | https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip   | Required for `preprocess_bird.py` |

Preprocessing:

```bash
# Extract dev.zip to data_raw/bird/
# Expected: dev.json, dev_databases/

python preprocess_bird.py
```

Output: `data/bird-dev/` with `bird-dev.json`, `bird-dev-nokg.json`, `golden.sql`, `database/`.

---

### Spider Test Suite (Alternative)

| Resource       | URL                                                                    | Notes |
| -------------- | ---------------------------------------------------------------------- | ----- |
| **Test suite** | https://drive.google.com/file/d/1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w/view | —     |

Use this to replace or overwrite the `spider-dev` processed folder, or to create a new test variant.

---

### Spider-DK

| Resource      | URL                               | Notes                      |
| ------------- | --------------------------------- | -------------------------- |
| **Spider-DK** | https://github.com/ygan/Spider-DK | Overwrites raw Spider data |

Preprocessing:

```bash
# Clone Spider-DK and place contents in data_raw/spider-dk/
# Expected: Spider-DK.json, database/

# Requires data_raw/spider/ (base Spider) to already exist
python preprocess_spider_dk.py
```

Output: `data/spider-dk-dev/`.

---

### Spider-Syn

| Resource       | URL                                | Notes                           |
| -------------- | ---------------------------------- | ------------------------------- |
| **Spider-Syn** | https://github.com/ygan/Spider-Syn | See `Spider-Syn/` subdirectory  |
| **Files**      | `dev.json`, `train_spider.json`    | Place in `data_raw/spider-syn/` |

Preprocessing:

```bash
# Place dev.json and train_spider.json in data_raw/spider-syn/
# Requires data_raw/spider/ to already exist

python preprocess_spider_syn.py
```

Output: `data/spider-syn-dev/`.

---

### Spider-Realistic

| Resource             | URL                                | Notes                     |
| -------------------- | ---------------------------------- | ------------------------- |
| **Spider-Realistic** | https://zenodo.org/records/5205322 | —                         |
| **File**             | `spider-realistic.json`            | Overwrite into raw Spider |

Preprocessing:

```bash
# Place spider-realistic.json in data_raw/spider-realistic/
# Requires data_raw/spider/ to already exist

python preprocess_spider_realistic.py
```

Output: `data/spider-realistic-dev/`.

---

## Output Format

Each processed JSON file contains records with:

| Field         | Description                     |
| ------------- | ------------------------------- |
| `id`          | Sample index                    |
| `db_id`       | Database identifier             |
| `instruction` | Full prompt (schema + question) |
| `output`      | Ground-truth SQL                |

The `golden.sql` file uses the packed format:

```
SELECT ... FROM table\t----- SQL-EVAL -----\tdb_id
```

This format is used for evaluation and for the `detailed.jsonl` output.
