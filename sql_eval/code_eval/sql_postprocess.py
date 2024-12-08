"""
Postprocess generations.json for evaluation.
Supports n_samples > 1: each query can have multiple inferences.
"""

import json
from pathlib import Path

# Must match sql_suites/utils.py
SPECIAL_SEPERATOR = "\t----- SQL-EVAL -----\t"

# Task name -> (db_root, dataset_path, mode)
# dataset_path: JSON with "output" and "db_id" per item (no golden.sql needed)
# Paths relative to project root (sql_eval/)
TASK_EVAL_CONFIG = {
    "spider-dev": ("sql_suites/data/spider-dev/database", "sql_suites/data/spider-dev/spider-dev.json", "spider"),
    "spider-train": (
        "sql_suites/data/spider-train/database",
        "sql_suites/data/spider-train/spider-train-filtered-6743.json",
        "spider",
    ),
    "spider-dev-chat": ("sql_suites/data/spider-dev/database", "sql_suites/data/spider-dev/spider-dev.json", "spider"),
    "spider-test": ("sql_suites/data/spider-test/database", "sql_suites/data/spider-test/spider-test.json", "spider"),
    "spider-dev-dk": (
        "sql_suites/data/spider-dk-dev/database",
        "sql_suites/data/spider-dk-dev/spider-dk-dev.json",
        "spider",
    ),
    "spider-dev-syn": (
        "sql_suites/data/spider-syn-dev/database",
        "sql_suites/data/spider-syn-dev/spider-syn-dev.json",
        "spider",
    ),
    "spider-dev-realistic": (
        "sql_suites/data/spider-realistic-dev/database",
        "sql_suites/data/spider-realistic-dev/spider-realistic-dev.json",
        "spider",
    ),
    "bird-dev": ("sql_suites/data/bird-dev/database", "sql_suites/data/bird-dev/bird-dev.json", "bird"),
    "bird-train": ("sql_suites/data/bird-train/database", "sql_suites/data/bird-train/bird-train.json", "bird"),
    "bird-dev-nokg": ("sql_suites/data/bird-dev/database", "sql_suites/data/bird-dev/bird-dev-nokg.json", "bird"),
}


def _fix_sql(line):
    """Strip and replace \\r/\\n with \\t."""
    return line.strip().replace("\r", "\t").replace("\n", "\t")


def _ensure_packed(line, db_id):
    """Append db_id if SPECIAL_SEPERATOR not in line."""
    if SPECIAL_SEPERATOR not in line:
        return f"{line}{SPECIAL_SEPERATOR}{db_id}"
    return line


def generations_to_sql(generations, fix=True, db_ids=None):
    """
    Convert generations to list of SQL strings (one per sample, flattened).
    For n_samples=1: returns [sql0, sql1, ...] one per query.
    For n_samples>1: returns [q0_s0, q0_s1, ..., q1_s0, ...] in row-major order.

    :param generations: [[gen1, gen2, ...], [gen1, ...], ...]  per query
    :param fix: If True, strip and replace \\r/\\n with \\t
    :param db_ids: optional list of db_id; when gen lacks SPECIAL_SEPERATOR, append it
    :return: list of (processed_sql, query_idx, sample_idx) tuples
    """
    result = []
    for q_idx, gens in enumerate(generations):
        db_id = db_ids[q_idx] if db_ids else None
        for s_idx, g in enumerate(gens):
            line = g
            if fix:
                line = _fix_sql(line)
            if db_id:
                line = _ensure_packed(line, db_id)
            result.append((line, q_idx, s_idx))
    return result


def generations_to_sql_list(generations, fix=True, db_ids=None):
    """
    Legacy: return flat list of SQL strings (first sample only per query).
    Used for n_samples=1 backward compatibility.
    """
    return [t[0] for t in generations_to_sql(generations, fix=fix, db_ids=db_ids) if t[2] == 0]


def save_generations_sql(generations, out_path, fix=True, db_ids=None):
    """
    Save generations to .sql file.
    - When n_samples=1: one SQL per line (packed format), compatible with eval.
    - When n_samples>1: do NOT save .sql (use generations.json for eval instead).
    """
    out_path = Path(out_path)
    flat = generations_to_sql(generations, fix=fix, db_ids=db_ids)
    n_samples = len(generations[0]) if generations else 0

    if n_samples == 1:
        lines = [t[0] for t in flat]
        with out_path.open("w") as f:
            for line in lines:
                f.write(f"{line}\n")
        print(f"generations.sql saved to {out_path}")
    else:
        # n_samples>1: don't save .sql, eval will use generations.json
        print(f"[n_samples={n_samples}>1] Skipping generations.sql (eval uses generations.json)")
    return out_path


def save_detailed_jsonl(
    idx_list,
    db_ids,
    inputs,
    raw_outputs_list,
    processed_sqls_flat,
    ground_truths,
    eval_details=None,
    out_path=None,
):
    """
    Save detailed per-sample results to jsonl.
    Each line = one (query_idx, sample_idx) pair.

    :param idx_list: list of query indices (序号)
    :param db_ids: list of db_id per query
    :param inputs: list of input prompts per query
    :param raw_outputs_list: list of lists - raw_outputs_list[q][s] = raw output for query q sample s
    :param processed_sqls_flat: list of (sql, query_idx, sample_idx) from generations_to_sql
    :param ground_truths: list of ground truth SQL per query
    :param eval_details: optional list of dicts from eval (query_idx, sample_idx, correct, status, ...)
    :param out_path: output jsonl path
    """
    out_path = Path(out_path)
    eval_by_key = {}
    if eval_details:
        for d in eval_details:
            k = (d["query_idx"], d["sample_idx"])
            eval_by_key[k] = d

    # Build sql lookup: (query_idx, sample_idx) -> processed_sql
    sql_by_key = {(q, s): sql for sql, q, s in processed_sqls_flat}

    with out_path.open("w", encoding="utf-8") as f:
        for q_idx, idx in enumerate(idx_list):
            for s_idx in range(len(raw_outputs_list[q_idx])):
                k = (q_idx, s_idx)
                row = {
                    "db_id": db_ids[q_idx],
                    "query_idx": idx,
                    "sample_idx": s_idx,
                    "input": inputs[q_idx],
                    "raw_output": raw_outputs_list[q_idx][s_idx],
                    "processed_sql": sql_by_key.get(k, ""),
                    "ground_truth": ground_truths[q_idx],
                }
                if k in eval_by_key:
                    ed = eval_by_key[k]
                    row["judge"] = {
                        "correct": ed.get("correct", False),
                        "status": ed.get("status", "unknown"),
                        "error_detail": ed.get("error_detail"),
                    }
                    row["error_reason"] = ed.get("error_detail") if ed.get("status") == "error" else None
                else:
                    row["judge"] = None
                    row["error_reason"] = None
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"detailed.jsonl saved to {out_path}")
