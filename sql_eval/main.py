import fnmatch
import json
from pathlib import Path

import datasets
import transformers
from transformers import HfArgumentParser

from code_eval import tasks
from code_eval.arguments import EvalArguments
from code_eval.evaluator import Evaluator
from code_eval.sql_postprocess import (
    TASK_EVAL_CONFIG,
    generations_to_sql,
    save_detailed_jsonl,
    save_generations_sql,
)
from code_eval.sql_eval_runner import run_sql_eval
from code_eval.tasks import ALL_TASKS


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False
        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def parse_args():
    parser = HfArgumentParser(EvalArguments)

    parser.add_argument(
        "--model",
        default="codeparrot/codeparrot-small",
        help="Model path or HuggingFace repo name",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        choices=MultiChoice(ALL_TASKS),
        help=f"Tasks from {ALL_TASKS}",
    )
    parser.add_argument(
        "--max_length_input",
        type=int,
        default=512,
        help="Maximum length of input",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum length of output",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--limit_start",
        type=int,
        default=0,
        help="Offset to start from",
    )
    parser.add_argument(
        "--postprocess",
        action="store_false",
        help="Postprocess model outputs (default: on)",
    )
    parser.add_argument(
        "--save_generations_path",
        type=str,
        default="generations.json",
        help="Path to save generations",
    )
    parser.add_argument(
        "--load_generations_path",
        type=str,
        default=None,
        help="Path to load generations.json; if set, skip generation and run postprocess/eval only",
    )

    parser.add_argument(
        "--load_data_path",
        type=str,
        default=None,
        help="Path of additional data for tasks",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="prompt",
        help="Prompt type",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Model cache path",
    )
    parser.add_argument(
        "--chat_mode",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="vLLM tensor parallel size (number of GPUs)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run SQL evaluation after generation (requires db_root, dataset JSON for task)",
    )
    parser.add_argument(
        "--eval_num_workers",
        type=int,
        default=16,
        help="Number of workers for eval",
    )
    parser.add_argument(
        "--eval_timeout",
        type=float,
        default=30.0,
        help="Execution timeout per SQL (seconds)",
    )
    parser.add_argument(
        "--save_result_path",
        type=str,
        default=None,
        help="Path to save eval result json (when --eval)",
    )
    parser.add_argument(
        "--no_sql_fix",
        action="store_true",
        help="Disable SQL postprocess fix (strip, replace \\r/\\n with \\t)",
    )

    return parser.parse_args()


def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def main():
    args = parse_args()
    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()

    if args.tasks is None:
        task_names = ALL_TASKS
    else:
        task_names = pattern_match(args.tasks.split(","), ALL_TASKS)

    print(f"Selected Tasks: {task_names}")

    evaluator = Evaluator(args)
    project_root = Path(__file__).resolve().parent
    base_save_path = Path(args.save_generations_path)
    load_path = Path(args.load_generations_path) if args.load_generations_path else None

    if load_path is not None:
        if len(task_names) != 1:
            raise ValueError("--load_generations_path requires exactly one task (--tasks)")
        load_path = Path(load_path).resolve()
        if not load_path.exists():
            raise FileNotFoundError(f"Generations file not found: {load_path}")

    for task_name in task_names:
        if load_path is not None:
            # Load from file, skip generation
            with load_path.open("r") as fp:
                generations = json.load(fp)
            task = tasks.get_task(task_name, args)
            dataset = task.get_dataset()
            n_tasks = len(generations)
            limit_start = args.limit_start
            references = [task.get_reference(dataset[i]) for i in range(limit_start, limit_start + n_tasks)]
            inputs = [task.get_prompt(dataset[i]) for i in range(limit_start, limit_start + n_tasks)]
            db_ids = [dataset[i]["db_id"] for i in range(limit_start, limit_start + n_tasks)]
            raw_generations = generations  # use loaded as raw when loading (no pre-postprocess available)
            save_path = load_path
            print(f"Loaded {n_tasks} generations from {load_path}")
        else:
            generations, references, raw_generations, inputs, db_ids = evaluator.generate_text(task_name)

            # Save generations.json (per-task path when multiple tasks)
            if len(task_names) > 1:
                save_path = base_save_path.parent / f"{base_save_path.stem}_{task_name}{base_save_path.suffix}"
            else:
                save_path = base_save_path

            save_path.parent.mkdir(parents=True, exist_ok=True)
            with save_path.open("w") as fp:
                json.dump(generations, fp, indent=2)
            print(f"generations saved to {save_path}")

        n_samples = len(generations[0]) if generations else 0
        processed_sqls_flat = generations_to_sql(generations, fix=not args.no_sql_fix, db_ids=db_ids)

        sql_path = save_path.with_suffix(".sql")
        save_generations_sql(generations, sql_path, fix=not args.no_sql_fix, db_ids=db_ids)

        # Eval details (when --eval)
        eval_details = None
        if args.eval:
            if task_name not in TASK_EVAL_CONFIG:
                print(f"[WARNING] No eval config for task {task_name}, skipping eval")
            else:
                db_root, dataset_path, mode = TASK_EVAL_CONFIG[task_name]
                db_root = project_root / db_root
                dataset_path = project_root / dataset_path
                if not db_root.exists():
                    print(f"[WARNING] db_root not found: {db_root}, skipping eval")
                elif not dataset_path.exists():
                    print(f"[WARNING] dataset JSON not found: {dataset_path}, skipping eval")
                else:
                    result_path = args.save_result_path or save_path.parent / "result.json"
                    detailed_eval_path = save_path.parent / "eval_detailed.jsonl"

                    if n_samples == 1:
                        run_sql_eval(
                            gen_sqls_path=sql_path,
                            db_root=db_root,
                            gold_json_path=dataset_path,
                            mode=mode,
                            num_workers=args.eval_num_workers,
                            exec_time_out=args.eval_timeout,
                            save_to=result_path,
                            save_detailed_to=detailed_eval_path,
                            project_root=project_root,
                            no_sql_fix=args.no_sql_fix,
                        )
                    else:
                        # n_samples > 1: write eval input json and use gen_json
                        gen_eval_path = save_path.parent / "generations_eval.json"
                        with gen_eval_path.open("w") as fp:
                            json.dump({"generations": generations, "db_ids": db_ids}, fp, indent=2)
                        run_sql_eval(
                            gen_json_path=gen_eval_path,
                            db_root=db_root,
                            gold_json_path=dataset_path,
                            mode=mode,
                            num_workers=args.eval_num_workers,
                            exec_time_out=args.eval_timeout,
                            save_to=result_path,
                            save_detailed_to=detailed_eval_path,
                            project_root=project_root,
                            no_sql_fix=args.no_sql_fix,
                        )
                        gen_eval_path.unlink(missing_ok=True)

                    with detailed_eval_path.open("r", encoding="utf-8") as fp:
                        eval_details = [json.loads(line) for line in fp if line.strip()]
                    detailed_eval_path.unlink()

        # Save detailed.jsonl (one row per query per sample)
        n_tasks = len(generations)
        idx_list = list(range(args.limit_start, args.limit_start + n_tasks))
        detailed_path = save_path.parent / "detailed.jsonl"
        save_detailed_jsonl(
            idx_list=idx_list,
            db_ids=db_ids,
            inputs=inputs,
            raw_outputs_list=raw_generations,
            processed_sqls_flat=processed_sqls_flat,
            ground_truths=references,
            eval_details=eval_details,
            out_path=detailed_path,
        )


if __name__ == "__main__":
    main()
