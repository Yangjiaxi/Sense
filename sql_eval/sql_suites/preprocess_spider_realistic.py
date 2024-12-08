import json
import shutil

from utils import save_json, save_lines, load_json, load_lines, join_seperator, extract_create_table_prompt
from pathlib import Path

SCHEMA_EXAMPLE_ROWS = 3


def create_codex_prompt(e, db_root):
    db_id = e["db_id"]
    db_path = Path(db_root) / db_id / f"{db_id}.sqlite"
    prompt = extract_create_table_prompt(db_path, limit_value=SCHEMA_EXAMPLE_ROWS)
    prompt += "-- Using valid SQLite, answer the following questions for the tables provided above.\n"
    prompt += f"Question: {e['question']}\n"

    return prompt


def process_data(questions, db_root):
    processed = []
    for e in questions:
        item = {
            "id": len(processed),
            "db_id": e["db_id"],
            "instruction": create_codex_prompt(e, db_root),
            "output": e["query"],
        }
        processed.append(item)

    return processed


if __name__ == "__main__":
    #
    # 1. copy databases
    # 2. build `CREATE and Question` style dataset
    #
    # -> data/spider-realistic-dev
    #    - database/
    #    - spider-realistic-dev.json (SELECT_ROW=3)
    #      - id
    #      - db_id
    #      - instruction
    #      - output
    #    - golden.sql
    #
    # There's no re-order.
    #
    output_root = Path("data/spider-realistic-dev")
    output_root.mkdir(exist_ok=True, parents=True)

    spider_raw_folder = Path("data_raw/spider")
    spider_realistic_raw_folder = Path("data_raw/spider-realistic")

    spider_realistic_dev_question = load_json(spider_realistic_raw_folder / "spider-realistic.json")
    spider_databases = spider_raw_folder / "database"

    spider_realistic_dev_question_processed = process_data(spider_realistic_dev_question, spider_databases)
    save_json(spider_realistic_dev_question_processed, output_root / "spider-realistic-dev.json")

    spider_dev_golden = [join_seperator(e["output"], e["db_id"]) for e in spider_realistic_dev_question_processed]
    save_lines(spider_dev_golden, output_root / "golden.sql")

    spider_used_databases = output_root / "database"
    spider_used_databases.mkdir(parents=True, exist_ok=True)
    db_ids = set([e["db_id"] for e in spider_realistic_dev_question_processed])

    for db_id in db_ids:
        raw_db_folder = spider_databases / db_id
        db_folder = spider_used_databases / db_id
        if db_folder.exists() and db_folder.is_dir():
            print(f"Skip already done: {db_id}")
        else:
            shutil.copytree(raw_db_folder, db_folder)
            print(f"Copied {db_id} from `{raw_db_folder}` to `{db_folder}`")
