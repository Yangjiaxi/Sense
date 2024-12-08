"""
Filter train dataset by SQL executability.
Keeps only samples whose ground-truth SQL executes successfully (and optionally returns non-empty result).
Output: {stem}-filtered-{n}.json (e.g. spider-train-filtered-8651.json)
"""

import sqlite3
from pathlib import Path

from func_timeout import FunctionTimedOut, func_timeout
from tqdm.contrib.concurrent import process_map

from utils import load_json, save_json, join_seperator


def _execute_sql(sql, db_path):
    """Execute SQL and return (success, non_empty)."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA synchronous=OFF")
        cursor.execute(sql)
        rows = cursor.fetchall()
    return True, len(rows) > 0


def _run_single(packed):
    """Run single SQL check. Returns (idx, execution_ok, non_empty)."""
    ex, db_root, timeout = packed
    ex_id = ex["id"]
    sql = ex["output"]
    db_path = Path(db_root) / ex["db_id"] / f"{ex['db_id']}.sqlite"

    try:
        exec_ok, non_empty = func_timeout(timeout, _execute_sql, args=(sql, str(db_path)))
        return {"idx": ex_id, "execution": exec_ok, "non_empty": non_empty}
    except FunctionTimedOut:
        return {"idx": ex_id, "execution": False, "non_empty": False}
    except Exception:
        return {"idx": ex_id, "execution": False, "non_empty": False}


def filter_train_dataset(
    data_path,
    db_root,
    require_non_empty=False,
    num_workers=16,
    timeout=30.0,
    output_path=None,
):
    """
    Filter train dataset by SQL executability.

    :param data_path: Path to train JSON (e.g. spider-train.json)
    :param db_root: Path to database root (e.g. data/spider-train/database)
    :param require_non_empty: If True, also require non-empty result (for BIRD)
    :param num_workers: Parallel workers
    :param timeout: Per-SQL timeout (seconds)
    :param output_path: Output path. If None, uses {stem}-filtered-{n}.json next to data_path
    :return: (filtered_data, n_passed, n_total)
    """
    data_path = Path(data_path)
    db_root = Path(db_root)

    data = load_json(data_path)
    if not data:
        return [], 0, 0

    results = process_map(
        _run_single,
        [(ex, db_root, timeout) for ex in data],
        max_workers=num_workers,
        chunksize=10,
    )

    results = sorted(results, key=lambda x: x["idx"])
    data_sorted = sorted(data, key=lambda x: x["id"])

    collected = []
    for ex, ret in zip(data_sorted, results):
        assert ex["id"] == ret["idx"]
        if ret["execution"] and (not require_non_empty or ret["non_empty"]):
            collected.append(ex)

    n_total = len(data)
    n_passed = len(collected)

    if output_path is None:
        stem = data_path.stem  # e.g. spider-train
        output_path = data_path.parent / f"{stem}-filtered-{n_passed}.json"

    save_json(collected, output_path)
    # Also save golden.sql for eval (must match filtered dataset order)
    golden_path = output_path.with_suffix("").with_suffix(".sql")
    golden_lines = [join_seperator(ex["output"], ex["db_id"]) for ex in collected]
    with golden_path.open("w") as f:
        for line in golden_lines:
            f.write(f"{line}\n")
    print(f"Filtered: {n_passed}/{n_total} passed -> {output_path}")
    print(f"Golden for eval -> {golden_path}")

    return collected, n_passed, n_total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter train dataset by SQL executability")
    parser.add_argument("--data", type=str, required=True, help="Path to train JSON")
    parser.add_argument("--db_root", type=str, required=True, help="Path to database root")
    parser.add_argument("--require_non_empty", action="store_true", help="Require non-empty result (for BIRD)")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", type=str, default=None, help="Output path (default: {stem}-filtered-{n}.json)")
    args = parser.parse_args()

    filter_train_dataset(
        data_path=args.data,
        db_root=args.db_root,
        require_non_empty=args.require_non_empty,
        num_workers=args.num_workers,
        timeout=args.timeout,
        output_path=args.output,
    )
