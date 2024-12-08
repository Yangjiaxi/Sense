"""
SQL evaluation. Supports:
- --gen_sqls: .sql file (one SQL per query, n_samples=1)
- --gen_json: .json with {"generations": [[...], ...], "db_ids": [...]} for n_samples>1
"""

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

from func_timeout import FunctionTimedOut, func_timeout
from rich.console import Console
from rich.table import Table
from tqdm.contrib.concurrent import process_map

from comparator import result_eq
from utils import SPECIAL_SEPERATOR, read_packed_sql


def execute_sql(db_file, gen_sql, gold_sql, cmp_method="spider"):
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()

        cursor.execute(gen_sql)
        predicted_res = cursor.fetchall()

        cursor.execute(gold_sql)
        ground_truth_res = cursor.fetchall()

        res = 0
        if cmp_method == "spider":
            order_matters = "orderby" in re.sub(r"\s+", gold_sql.lower(), "")
            if result_eq(predicted_res, ground_truth_res, order_matters=order_matters):
                res = 1
        elif cmp_method == "bird":
            if set(predicted_res) == set(ground_truth_res):
                res = 1

    return res


def find_detail(reason):
    patterns = [
        "ambiguous column name",
        "no such column",
        "no such table",
        "no such function",
        "syntax error",
    ]
    for p in patterns:
        if p in reason:
            return p
    return "others"


def execute_model(packed):
    db_file, gen_sql, gold_sql, exec_time_out, mode = packed

    status = "failed"
    detail = None
    try:
        res = func_timeout(exec_time_out, execute_sql, args=(db_file, gen_sql, gold_sql, mode))
        status = "success"
    except FunctionTimedOut:
        status = "timeout"
        res = 0
    except Exception as e:
        detail = find_detail(str(e))
        status = "error"
        res = 0

    return {"res": res, "status": status, "detail": detail}


def run_sqls_parallel(packed_payload, mode, num_workers=64, exec_time_out=30.0):
    return process_map(
        execute_model,
        [(*payload, exec_time_out, mode) for payload in packed_payload],
        max_workers=num_workers,
        chunksize=10,
    )


def mean(arr):
    return sum(arr) / len(arr) if arr else 0.0


def load_gold_from_json(dataset_path, db_root):
    """
    Load ground truth from dataset JSON. Each item must have "output" (gold SQL) and "db_id".
    Returns (gold_sqls, gold_dbs) matching read_packed_sql format.
    """
    with Path(dataset_path).open("r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    gold_sqls = [item["output"] for item in data]
    db_ids = [item["db_id"] for item in data]
    db_root = Path(db_root)
    gold_dbs = [str(db_root / db_id / f"{db_id}.sqlite") for db_id in db_ids]
    print(f"Load {len(gold_sqls)} gold SQLs from {dataset_path}")
    return gold_sqls, gold_dbs


def load_gen_from_json(gen_json_path, db_root, fix=True):
    """
    Load generations from JSON. Expected format:
    {"generations": [[gen1, gen2, ...], ...], "db_ids": ["db1", "db2", ...]}
    Returns: list of (db_file, gen_sql, gold_sql, query_idx, sample_idx)
    """
    with Path(gen_json_path).open("r") as f:
        data = json.load(f)
    generations = data["generations"]
    db_ids = data["db_ids"]
    n_queries = len(generations)
    assert len(db_ids) == n_queries

    def _fix(s):
        return s.strip().replace("\r", "\t").replace("\n", "\t") if fix else s

    def _pack(sql, db_id):
        if SPECIAL_SEPERATOR not in sql:
            return f"{sql}{SPECIAL_SEPERATOR}{db_id}"
        return sql

    flat = []
    for q_idx, gens in enumerate(generations):
        db_id = db_ids[q_idx]
        db_file = Path(db_root) / db_id / f"{db_id}.sqlite"
        for s_idx, g in enumerate(gens):
            sql = _pack(_fix(g), db_id) if fix else g
            flat.append((str(db_file), sql, q_idx, s_idx))
    return flat


def compute_score(exec_results, multi_sample=False, n_samples_per_query=None):
    """
    multi_sample: if True, aggregate by query (exec_results grouped by query_idx).
    n_samples_per_query: list of n_samples for each query (when multi_sample).
    """
    error_types = defaultdict(lambda: 0)
    for exec_ret in exec_results:
        if exec_ret.get("detail"):
            error_types[exec_ret["detail"]] += 1
    error_types = dict(error_types)
    for k, v in error_types.items():
        print(f"\t{k}: {v}")

    if not multi_sample:
        exec_dict = {"all": [1 if r["status"] == "success" else 0 for r in exec_results]}
        acc_dict = {"all": [r["res"] for r in exec_results]}
        num_dict = {"all": len(exec_results)}
        acc_score = {k: mean(v) for k, v in acc_dict.items()}
        exec_score = {k: mean(v) for k, v in exec_dict.items()}
        return acc_score, exec_score, num_dict, error_types

    # Multi-sample: aggregate by query
    # exec_results have (query_idx, sample_idx) in metadata
    query_results = defaultdict(list)
    for item in exec_results:
        q = item["query_idx"]
        query_results[q].append(item)

    n_queries = len(query_results)
    exec_any = []  # 1 if any sample executable
    acc_any = []  # 1 if any sample correct (execution accuracy)
    exec_avg = []
    acc_avg = []
    for q in sorted(query_results.keys()):
        items = query_results[q]
        exec_any.append(1 if any(r["status"] == "success" for r in items) else 0)
        acc_any.append(1 if any(r["res"] == 1 for r in items) else 0)
        exec_avg.append(mean([1 if r["status"] == "success" else 0 for r in items]))
        acc_avg.append(mean([r["res"] for r in items]))

    acc_score = {
        "all": mean(acc_any),
        "all_avg": mean(acc_avg),
    }
    exec_score = {
        "all": mean(exec_any),
        "all_avg": mean(exec_avg),
    }
    num_dict = {"all": n_queries, "total_samples": sum(len(v) for v in query_results.values())}
    return acc_score, exec_score, num_dict, error_types


def print_scores(acc_score, exec_score, num_dict, multi_sample=False):
    to_percent = lambda num: f"{num:.2%}"
    table = Table(show_footer=True)
    table.add_column("Part", footer="All", justify="right")
    table.add_column("Executable", footer=to_percent(exec_score["all"]), justify="right")
    table.add_column("Accuracy", footer=to_percent(acc_score["all"]), justify="right")
    if multi_sample:
        table.add_column("Exec(avg)", footer=to_percent(exec_score.get("all_avg", 0)), justify="right")
        table.add_column("Acc(avg)", footer=to_percent(acc_score.get("all_avg", 0)), justify="right")
    Console().print(table)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--gen_sqls", type=str, default=None, help="Path to .sql file (n_samples=1)")
    args_parser.add_argument(
        "--gen_json", type=str, default=None, help="Path to .json with generations+db_ids (n_samples>=1)"
    )
    args_parser.add_argument(
        "--gold_json",
        type=str,
        required=True,
        help="Path to dataset JSON with 'output' and 'db_id' per item (ground truth)",
    )
    args_parser.add_argument("--db_root", type=str, required=True)
    args_parser.add_argument("--num_workers", type=int, default=1)
    args_parser.add_argument("--exec_time_out", type=float, default=30.0)
    args_parser.add_argument("--mode", type=str, required=True, choices=["bird", "spider"])
    args_parser.add_argument("--save_to", type=str, default=None)
    args_parser.add_argument("--save_detailed", type=str, default=None, help="Save per-sample results to jsonl")
    args_parser.add_argument("--no_sql_fix", action="store_true", help="Disable SQL fix when loading from gen_json")
    args_parser.add_argument("--debug", action="store_true")
    args = args_parser.parse_args()

    if (args.gen_sqls is None) == (args.gen_json is None):
        raise ValueError("Exactly one of --gen_sqls or --gen_json is required")

    db_root = Path(args.db_root)
    gold_sqls, gold_dbs = load_gold_from_json(args.gold_json, db_root=args.db_root)
    n_queries_full = len(gold_sqls)

    multi_sample = False
    packed_payload = []
    query_sample_indices = []  # (query_idx, sample_idx) for each payload item

    if args.gen_sqls:
        gen_sqls, gen_dbs = read_packed_sql(args.gen_sqls, db_root=args.db_root)
        n_gen = len(gen_sqls)
        # When gen is a subset (e.g. limit 32), slice gold to match
        if n_gen < n_queries_full:
            gold_sqls = gold_sqls[:n_gen]
            gold_dbs = gold_dbs[:n_gen]
        n_queries = len(gold_sqls)
        if not args.debug:
            assert len(gen_dbs) == len(gold_dbs), f"{len(gen_dbs) = } != {len(gold_dbs) = }"
            assert len(gen_sqls) == len(gold_sqls)
        for i, (db_file, gen_sql, gold_sql) in enumerate(zip(gold_dbs, gen_sqls, gold_sqls)):
            packed_payload.append((db_file, gen_sql, gold_sql))
            query_sample_indices.append((i, 0))
    else:
        flat = load_gen_from_json(args.gen_json, args.db_root, fix=not args.no_sql_fix)
        n_gen_flat = len(flat)
        n_queries = max((q for _, _, q, _ in flat), default=-1) + 1 if flat else 0
        if n_queries < n_queries_full:
            gold_sqls = gold_sqls[:n_queries]
            gold_dbs = gold_dbs[:n_queries]
        multi_sample = n_gen_flat > n_queries
        # flat: (db_file, gen_sql, query_idx, sample_idx)
        # We need gold_sql per (query_idx, sample_idx)
        for db_file, gen_sql, q_idx, s_idx in flat:
            gold_sql = gold_sqls[q_idx]
            gold_db = gold_dbs[q_idx]
            packed_payload.append((db_file, gen_sql, gold_sql))
            query_sample_indices.append((q_idx, s_idx))

    exec_result = run_sqls_parallel(
        packed_payload,
        mode=args.mode,
        num_workers=args.num_workers,
        exec_time_out=args.exec_time_out,
    )

    # Attach (query_idx, sample_idx) to each result
    for i, (q, s) in enumerate(query_sample_indices):
        exec_result[i]["query_idx"] = q
        exec_result[i]["sample_idx"] = s

    if args.save_detailed:
        with Path(args.save_detailed).open("w") as f:
            for (db_file, gen_sql, gold_sql), ret in zip(packed_payload, exec_result):
                db_id = Path(db_file).parent.name
                line = (
                    json.dumps(
                        {
                            "query_idx": ret["query_idx"],
                            "sample_idx": ret["sample_idx"],
                            "db_id": db_id,
                            "gen_sql": gen_sql,
                            "gold_sql": gold_sql,
                            "correct": ret["res"] == 1,
                            "status": ret["status"],
                            "error_detail": ret["detail"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                f.write(line)
        print(f"Detailed results saved to {args.save_detailed}")

    acc_score, exec_score, num_dict, error_counter = compute_score(exec_result, multi_sample=multi_sample)
    print_scores(acc_score, exec_score, num_dict, multi_sample=multi_sample)

    n_samples_per_query = len(exec_result) // n_queries if n_queries else 0
    metric_dict = {
        "acc": acc_score,
        "exec": exec_score,
        "nums": num_dict,
        "error_counter": error_counter,
        "n_samples_per_query": n_samples_per_query,
    }

    if args.save_to is not None:
        with Path(args.save_to).open("w") as f:
            json.dump(metric_dict, f, indent=2)
