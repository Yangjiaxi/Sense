from code_eval.base import Task
from datasets import load_dataset

SPECIAL_SEPERATOR = "\t----- SQL-EVAL -----\t"


class SqlTaskChatBase(Task):
    def __init__(self):
        super().__init__(
            stop_words=[
                # "\n\n",
                # ";",
                "<pad>",
                "<|endoftext|>",
                "</s>",
                "<|im_end|>",
            ],
            requires_execution=False,
        )
        self.datasets = self.dataset["train"].to_list()
        print(f"{len(self.datasets) = }")

    def get_dataset(self):
        return self.datasets

    def get_prompt(self, doc):
        return doc["instruction"] + "\nOutput your SQL query starts with keyword `SELECT`."

    def get_reference(self, doc):
        return doc["output"]

    @staticmethod
    def _stop_at_stop_token(decoded_string, stop_tokens):
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return decoded_string[:min_stop_index]

    def postprocess_generation(self, generation, idx, prompt_stripped=False, input_str_len=None):
        if input_str_len is not None:
            input_len = input_str_len
        else:
            doc = self.datasets[idx]
            input_len = doc.get("input_str_len", None)
        if input_len is None:
            return ""
        generation = generation[input_len:]
        generation = self._stop_at_stop_token(generation, self.stop_words)

        return generation

    def process_results(self, generations, references):
        pass
