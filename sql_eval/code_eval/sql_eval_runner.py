"""
Run sql_suites/eval.py for SQL evaluation.
Supports n_samples=1 (gen_sqls_path) or n_samples>1 (gen_json_path with generations+db_ids).
"""

import subprocess
import sys
from pathlib import Path


def run_sql_eval(
    gen_sqls_path=None,
    gen_json_path=None,
    db_root=None,
    gold_json_path=None,
    mode="spider",
    num_workers=16,
    exec_time_out=30.0,
    save_to=None,
    save_detailed_to=None,
    project_root=None,
    no_sql_fix=False,
):
    """
    Run sql_suites/eval.py via subprocess.

    :param gen_sqls_path: Path to .sql file (n_samples=1). Mutually exclusive with gen_json_path.
    :param gen_json_path: Path to .json with {"generations": [[...], ...], "db_ids": [...]} (n_samples>=1).
    :param project_root: Project root (sql_eval/). If None, use parent of code_eval.
    :param save_detailed_to: Path to save per-sample eval results (jsonl).
    """
    if (gen_sqls_path is None) == (gen_json_path is None):
        raise ValueError("Exactly one of gen_sqls_path or gen_json_path is required")

    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    project_root = Path(project_root)
    db_root = Path(db_root).resolve()
    gold_json_path = Path(gold_json_path).resolve()

    cmd = [
        sys.executable,
        "-u",
        str(project_root / "sql_suites" / "eval.py"),
        "--gold_json",
        str(gold_json_path),
        "--db_root",
        str(db_root),
        "--mode",
        mode,
        "--num_workers",
        str(num_workers),
        "--exec_time_out",
        str(exec_time_out),
    ]
    if gen_sqls_path is not None:
        cmd.extend(["--gen_sqls", str(Path(gen_sqls_path).resolve())])
    else:
        cmd.extend(["--gen_json", str(Path(gen_json_path).resolve())])
        if no_sql_fix:
            cmd.append("--no_sql_fix")

    if save_to:
        cmd.extend(["--save_to", str(Path(save_to).resolve())])
    if save_detailed_to:
        cmd.extend(["--save_detailed", str(Path(save_detailed_to).resolve())])

    subprocess.run(cmd, cwd=str(project_root), check=True)
