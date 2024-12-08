from .sql_task_base import SqlTaskBase


class SpiderTrain(SqlTaskBase):
    # Filtered by train_filter.py (executable SQL only); use as eval data provider

    # DATASET_PATH = "sql_suites/data/spider-train/spider-train.json"  # original, unfiltered
    DATASET_PATH = "sql_suites/data/spider-train/spider-train-filtered-6743.json"

    DATASET_NAME = None
