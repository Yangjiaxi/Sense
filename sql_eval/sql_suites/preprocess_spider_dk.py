import json
import shutil

from utils import save_json, save_lines, load_json, load_lines, join_seperator, extract_create_table_prompt
from pathlib import Path

SCHEMA_EXAMPLE_ROWS = 3


def create_codex_prompt(e, db_root, db_root_new):
    db_id = e["db_id"]

    db_path = Path(db_root) / db_id / f"{db_id}.sqlite"
    if not db_path.exists():
        db_path = Path(db_root_new) / db_id / f"{db_id}.sqlite"

    prompt = extract_create_table_prompt(db_path, limit_value=SCHEMA_EXAMPLE_ROWS)
    prompt += "-- Using valid SQLite, answer the following questions for the tables provided above.\n"
    prompt += f"Question: {e['question']}\n"

    return prompt


def process_data(questions, db_root, db_root_new):
    processed = []
    for e in questions:
        item = {
            "id": len(processed),
            "db_id": e["db_id"],
            "instruction": create_codex_prompt(e, db_root, db_root_new),
            "output": e["query"],
        }
        processed.append(item)

    return processed


if __name__ == "__main__":
    #
    # 1. copy databases
    # 2. build `CREATE and Question` style dataset
    #
    # -> data/spider-dk-dev
    #    - database/
    #    - spider-dk-dev.json (SELECT_ROW=3)
    #      - id
    #      - db_id
    #      - instruction
    #      - output
    #    - golden.sql
    #
    # There's no re-order.
    #
    output_root = Path("data/spider-dk-dev")
    output_root.mkdir(exist_ok=True, parents=True)

    spider_raw_folder = Path("data_raw/spider")
    spider_dk_raw_folder = Path("data_raw/spider-dk")

    spider_dk_dev_question = load_json(spider_dk_raw_folder / "Spider-DK.json")
    spider_databases = spider_raw_folder / "database"
    spdier_dk_databases = spider_dk_raw_folder / "database"

    spider_dk_dev_question_processed = process_data(spider_dk_dev_question, spider_databases, spdier_dk_databases)
    save_json(spider_dk_dev_question_processed, output_root / "spider-dk-dev.json")

    spider_dev_golden = [join_seperator(e["output"], e["db_id"]) for e in spider_dk_dev_question_processed]
    save_lines(spider_dev_golden, output_root / "golden.sql")

    spider_dk_used_databases = output_root / "database"
    spider_dk_used_databases.mkdir(parents=True, exist_ok=True)
    db_ids = set([e["db_id"] for e in spider_dk_dev_question_processed])

    for db_id in db_ids:
        raw_db_folder = spider_databases / db_id
        dk_new_db_folder = spdier_dk_databases / db_id

        db_folder = spider_dk_used_databases / db_id
        if db_folder.exists() and db_folder.is_dir():
            print(f"Skip already done: {db_id}")
        else:
            if raw_db_folder.exists():
                source_db_folder = raw_db_folder
            elif dk_new_db_folder.exists():
                source_db_folder = dk_new_db_folder
            else:
                raise ValueError(f"Not found db: {db_id}")

            shutil.copytree(source_db_folder, db_folder)
            print(f"Copied {db_id} from `{source_db_folder}` to `{db_folder}`")
