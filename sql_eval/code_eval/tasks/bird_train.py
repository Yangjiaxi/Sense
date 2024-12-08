from .sql_task_base import SqlTaskBase


class BirdTrain(SqlTaskBase):
    DATASET_PATH = "sql_suites/data/bird-train/bird-train.json"
    DATASET_NAME = None
