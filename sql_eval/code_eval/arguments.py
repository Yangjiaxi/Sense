from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalArguments:
    """
    Configuration for running the evaluation.
    """

    prefix: Optional[str] = field(
        default="",
        metadata={"help": "Prefix to add to the prompt. For example InCoder needs prefix='<| file ext=.py |>\n'"},
    )
    do_sample: Optional[bool] = field(
        default=True,
        metadata={"help": "Sample from the language model's output distribution."},
    )
    greedy: Optional[bool] = field(
        default=False,
        metadata={"help": "Use greedy decoding (temperature=0, deterministic). Overrides temperature."},
    )
    temperature: Optional[float] = field(
        default=0.2,
        metadata={"help": "Sampling temperature (only when not greedy). 0=greedy, >0=sampling."},
    )
    top_k: Optional[int] = field(default=0, metadata={"help": "Top-k parameter used for generation."})
    top_p: Optional[float] = field(default=0.95, metadata={"help": "Top-p parameter used for nucleus sampling."})
    # top_k: Optional[int] = field(default=None, metadata={"help": "Top-k parameter used for generation."})
    # top_p: Optional[float] = field(default=None, metadata={"help": "Top-p parameter used for nucleus sampling."})
    n_samples: Optional[int] = field(
        default=1,
        metadata={"help": "Number of completions to generate for each sample."},
    )
    num_beams: Optional[int] = field(default=1)  # beam search
    eos: Optional[str] = field(default="<|endoftext|>", metadata={"help": "end of sentence token."})
    seed: Optional[int] = field(default=0, metadata={"help": "Random seed used for evaluation."})

    # do_sample = False
    # num_beams = 4
    # num_return_sequences = 4
