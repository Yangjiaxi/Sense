#!/bin/bash
# Generate DPO pairs from exec results.
# Output: {model}.{dataset}.grouped-any-limit1.json (e.g. deepseek-1.3b.bird.grouped-any-limit1.json)

cd "$(dirname "$0")"

SCRIPT_DIR="$(pwd)"
DATA_ROOT="${SCRIPT_DIR}/../../data"

# deepseek-1.3b
python construct_dpo_pairs.py \
  ${DATA_ROOT}/sampling/deepseek-1.3b-after.sft-grouped/bird-exec.json \
  --output ${DATA_ROOT}/dpo/deepseek-1.3b.bird.grouped-any-limit1.json

python construct_dpo_pairs.py \
  ${DATA_ROOT}/sampling/deepseek-1.3b-after.sft-grouped/spider-exec.json \
  --output ${DATA_ROOT}/dpo/deepseek-1.3b.spider.grouped-any-limit1.json
