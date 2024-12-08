import warnings

from code_eval import tasks
from code_eval.generation import parallel_generations


class Evaluator:
    def __init__(self, args):
        self.args = args

    def generate_text(self, task_name):
        task = tasks.get_task(task_name, self.args)
        dataset = task.get_dataset()
        n_tasks = self.args.limit if self.args.limit else len(dataset)

        generations, raw_generations = parallel_generations(
            task,
            dataset,
            n_tasks=n_tasks,
            args=self.args,
        )
        if self.args.do_sample:
            expected_generated = self.args.n_samples
        else:
            expected_generated = self.args.num_beams

        if len(generations[0]) > expected_generated:
            generations = [l[:expected_generated] for l in generations]
            raw_generations = [l[:expected_generated] for l in raw_generations]
            warnings.warn(
                f"Removing extra predictions from {len(generations[0])} to only keep nsamples={expected_generated}"
            )

        references = [
            task.get_reference(dataset[i]) for i in range(self.args.limit_start, self.args.limit_start + n_tasks)
        ]
        inputs = [task.get_prompt(dataset[i]) for i in range(self.args.limit_start, self.args.limit_start + n_tasks)]
        db_ids = [dataset[i]["db_id"] for i in range(self.args.limit_start, self.args.limit_start + n_tasks)]
        return generations, references, raw_generations, inputs, db_ids
