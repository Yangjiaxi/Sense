"""
vLLM offline inference for code generation.
Model loading with TP, synchronous batch inference, stopwords and postprocess adapted.
"""

from vllm import LLM
from vllm.sampling_params import SamplingParams

from code_eval.utils import build_prompts_for_vllm


def parallel_generations(task, dataset, n_tasks, args):
    """
    Generate using vLLM: load model with TP, run synchronous offline inference.
    """
    # Build stop words: task.stop_words + optional eos_token
    stop_words = list(task.stop_words) if task.stop_words else []

    # Load vLLM model with TP
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    # Add eos_token to stop if present
    if tokenizer.eos_token and tokenizer.eos_token not in stop_words:
        stop_words.append(tokenizer.eos_token)

    # Filter empty stop strings (vLLM rejects them)
    stop_words = [s for s in stop_words if s]

    # Sampling params: greedy (temp=0) or temperature sampling
    use_greedy = getattr(args, "greedy", False) or (not getattr(args, "do_sample", True))
    temperature = 0.0 if use_greedy else getattr(args, "temperature", 0.2)
    top_p = 1.0 if use_greedy else getattr(args, "top_p", 0.95)
    top_k = -1 if use_greedy else (getattr(args, "top_k", 0) if getattr(args, "top_k", 0) > 0 else -1)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=args.max_new_tokens,
        stop=stop_words if stop_words else None,
        n=args.n_samples,
        skip_special_tokens=True,
    )

    # Build prompts
    prompts, extra_info = build_prompts_for_vllm(
        task,
        dataset,
        limit_start=args.limit_start,
        n_tasks=n_tasks,
        prefix=args.prefix,
        chat_mode=args.chat_mode,
        tokenizer=tokenizer,
        max_length=args.max_length_input,
    )

    input_str_lens = extra_info.get("input_str_lens", [0] * n_tasks)
    if not input_str_lens:
        input_str_lens = [0] * n_tasks

    print(f"number of problems for this task is {n_tasks}")
    if args.chat_mode:
        print("Chat-Mode enabled.")
    else:
        print("Completion mode (default)")
    print(f"Sampling: {'greedy' if use_greedy else f'temperature={temperature}, top_p={top_p}, top_k={top_k}'}")

    # Synchronous offline inference
    outputs = llm.generate(prompts, sampling_params)

    # Collect generations: outputs[i].outputs[j].text for task i, sample j
    code_gens = [[] for _ in range(n_tasks)]
    raw_gens = [[] for _ in range(n_tasks)]
    for i, request_output in enumerate(outputs):
        for j, output in enumerate(request_output.outputs):
            gen_code = output.text  # raw model output
            raw_gens[i].append(gen_code)
            if args.postprocess:
                # vLLM returns only generated part; prompt_stripped=True, input_str_len=0 for chat
                inp_len = input_str_lens[i] if args.chat_mode else 0
                processed = task.postprocess_generation(
                    gen_code,
                    int(i) + args.limit_start,
                    prompt_stripped=True,
                    input_str_len=inp_len,
                )
                code_gens[i].append(processed)
            else:
                code_gens[i].append(gen_code)

    return code_gens, raw_gens
