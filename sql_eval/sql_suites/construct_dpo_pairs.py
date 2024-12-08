"""
Construct DPO pairs from exec results.
Output: {model}.{dataset}.grouped-any-limit1.json (e.g. deepseek-1.3b.bird.grouped-any-limit1.json)

Fixed params (no grid search):
  - dedup=True: treat a group as one sample
  - neg_filter=all: use all negative samples
  - keep_gold=False: exclude gold (GT) from positive samples
  - neg_limit=1: limit to 1 negative per positive
"""

import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm

from utils import load_json, save_json


def construct_dpo_pairs(data, dedup=True, neg_filter="all", keep_gold=False, neg_limit=1):
    """Construct DPO pairs from exec result data."""
    collected_pairs = []
    for ex in tqdm(data):
        banned_keys = ["filtered_id", "system", "output"]
        for k in banned_keys:
            if k in ex:
                ex.pop(k)

        sampling = ex.pop("sampling")

        if dedup:
            deduped = {e["group"]: e for e in sampling}
            sampling = list(deduped.values())

        if not keep_gold:
            sampling = [e for e in sampling if e["group"] != "GT"]
        if neg_filter == "exec":
            sampling = [e for e in sampling if e["execution"]]

        pos_sqls = list(filter(lambda e: e["result"], sampling))
        neg_sqls = list(filter(lambda e: not e["result"], sampling))

        assert len(pos_sqls) + len(neg_sqls) == len(
            sampling
        ), f"{len(pos_sqls)=} + {len(neg_sqls)=} != {len(sampling)=}"

        if neg_limit:
            factor = neg_limit
            if len(neg_sqls) > factor * len(pos_sqls):
                if len(neg_sqls) > factor:
                    neg_sqls = random.sample(neg_sqls, factor)

        for pos in pos_sqls:
            for neg in neg_sqls:
                collected_pairs.append(
                    {
                        "pair_id": len(collected_pairs),
                        **ex,
                        "chosen": pos["sql"],
                        "rejected": neg["sql"],
                    }
                )

    return collected_pairs


def infer_output_filename(input_path):
    """
    Infer output filename from input path.
    e.g. data/sampling/deepseek-1.3b-after.sft-grouped/bird-exec.json
    -> deepseek-1.3b.bird.grouped-any-limit1.json
    """
    path = Path(input_path)
    parent_name = path.parent.name  # e.g. deepseek-1.3b-after.sft-grouped
    stem = path.stem  # e.g. bird-exec

    # model: deepseek-1.3b (strip -after.sft-grouped or similar)
    model = parent_name.split("-after")[0] if "-after" in parent_name else parent_name.split(".")[0]

    # dataset: bird or spider (strip -exec)
    dataset = stem.replace("-exec", "") if stem.endswith("-exec") else stem

    return f"{model}.{dataset}.grouped-any-limit1.json"


if __name__ == "__main__":
    random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Input exec result JSON (e.g. bird-exec.json, spider-exec.json)")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output path. Default: same dir as input, named {model}.{dataset}.grouped-any-limit1.json",
    )
    args = parser.parse_args()

    file = Path(args.file)
    if not file.exists():
        raise ValueError(f"Not exist: {file}")

    data = load_json(file)
    collected_pairs = construct_dpo_pairs(
        data,
        dedup=True,
        neg_filter="all",
        keep_gold=False,
        neg_limit=1,
    )

    print(f"Total: {len(collected_pairs)}")

    if args.output:
        save_path = Path(args.output)
    else:
        save_path = file.parent / infer_output_filename(file)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(collected_pairs, save_path)
