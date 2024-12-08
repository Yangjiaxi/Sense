from pprint import pprint

from . import (
    spider_dev,
    spider_test,
    spider_train,
    bird_dev,
    bird_train,
    spider_dev_chat,
)

TASK_REGISTRY = {
    "spider-dev": spider_dev.SpiderDev,
    "spider-train": spider_train.SpiderTrain,
    "spider-dev-dk": spider_dev.SpiderDevDk,
    "spider-dev-syn": spider_dev.SpiderDevSyn,
    "spider-dev-realistic": spider_dev.SpiderDevRealistic,
    "spider-test": spider_test.SpiderTest,
    "bird-dev": bird_dev.BirdDev,
    "bird-train": bird_train.BirdTrain,
    "bird-dev-nokg": bird_dev.BirdDevNoKG,
    # ----- chat model -----
    "spider-dev-chat": spider_dev_chat.SpiderDevChat,
}

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name, args=None):
    try:
        kwargs = {}
        return TASK_REGISTRY[task_name](**kwargs)
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
