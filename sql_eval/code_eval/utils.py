"""
Utilities for vLLM inference: prompt building and postprocess helpers.
"""


def build_prompts_for_vllm(task, dataset, limit_start, n_tasks, prefix="", chat_mode=False, tokenizer=None, max_length=2048):
    """
    Build list of prompt strings for vLLM offline inference.

    :param task: Task instance
    :param dataset: Dataset
    :param limit_start: Start index
    :param n_tasks: Number of tasks
    :param prefix: Optional prefix for completion mode
    :param chat_mode: If True, apply chat template
    :param tokenizer: Required for chat_mode - used for apply_chat_template
    :param max_length: Max length for chat template truncation
    :return: (prompts: list[str], extra_info: dict) - extra_info has input_str_lens for chat mode
    """
    prompts = []
    input_str_lens = []

    for sample in range(limit_start, limit_start + n_tasks):
        doc = dataset[sample]
        prompt_contents = task.get_prompt(doc)

        if chat_mode:
            if not tokenizer:
                raise ValueError("tokenizer is required for chat_mode")
            if isinstance(prompt_contents, str):
                messages = [{"role": "user", "content": prompt_contents}]
            else:
                raise ValueError(f"Unsupported prompt format for chat: {type(prompt_contents)}")

            result = tokenizer.apply_chat_template(
                messages,
                padding=False,
                truncation=True,
                max_length=max_length,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            # Handle BatchEncoding (input_ids) or raw tensor
            if hasattr(result, "input_ids"):
                tokens = result.input_ids
            else:
                tokens = result
            prompt_str = tokenizer.decode(tokens[0], skip_special_tokens=False)
            input_str_lens.append(len(prompt_str))
            prompts.append(prompt_str)
        else:
            # Completion mode
            if isinstance(prompt_contents, str):
                prompt = prefix + prompt_contents
            elif isinstance(prompt_contents, dict):
                if set(prompt_contents.keys()) == {"prefix", "suffix"}:
                    raise NotImplementedError("Infilling mode not supported in vLLM path")
                elif set(prompt_contents.keys()) == {"instruction", "context"}:
                    raise NotImplementedError("Instruction mode not supported in vLLM path")
                else:
                    raise ValueError(f"Unsupported prompt format: {type(prompt_contents)}")
            else:
                raise ValueError(f"Unsupported prompt format: {type(prompt_contents)}")
            prompts.append(prompt)

    extra_info = {"input_str_lens": input_str_lens} if chat_mode else {}
    return prompts, extra_info
