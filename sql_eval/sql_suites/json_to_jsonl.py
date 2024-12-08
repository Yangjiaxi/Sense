#!/usr/bin/env python3
"""
Convert .json to .jsonl with validation.
Check: JSON is list of dict, all dicts have same fields.
"""

import argparse
import json
from pathlib import Path


def check_and_convert(json_path, jsonl_path=None):
    with Path(json_path).open("r") as f:
        data = json.load(f)

    # Check: must be list
    if not isinstance(data, list):
        raise ValueError(f"Expected list, got {type(data).__name__}")

    # Check: all elements must be dict
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item[{i}] is {type(item).__name__}, expected dict")

    # Check: all dicts have same fields
    if data:
        ref_keys = frozenset(data[0].keys())
        for i, item in enumerate(data[1:], start=1):
            if frozenset(item.keys()) != ref_keys:
                diff = set(item.keys()) ^ ref_keys
                raise ValueError(f"Item[{i}] has different keys: {diff}")

    out = Path(jsonl_path) if jsonl_path else Path(json_path).with_suffix(".jsonl")
    with out.open("w") as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Check OK: list of {len(data)} dicts, keys={sorted(ref_keys)}")
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="Input .json file")
    parser.add_argument("-o", "--output", default=None, help="Output .jsonl path (default: same name with .jsonl)")
    args = parser.parse_args()

    check_and_convert(args.json_file, args.output)
