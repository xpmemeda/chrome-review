import tqdm
import numpy
import json
import sys
import argparse
from typing import Any, Optional, Union, List, Dict, Tuple
from transformers import AutoTokenizer

from suite.datasets import Dataset, StdMessageJsonL
from suite.tokenizer import Tokenizer
from suite.comm.radixtree import RadixTree

Messages = List[Dict[str, str]]
SamplingParams = Dict[str, Any]


class DatasetProfiler:
    def __init__(
        self, dataset: Dataset, tokenizer: Tokenizer, num_reqs: int, radix_size: int
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.radix_tree = RadixTree(radix_size)

        self.prefix_hit_size: List[Tuple[int, int]] = []

        if num_reqs > dataset.size():
            raise RuntimeError("specified num requests > datasets requests.")
        if num_reqs == 0:
            tqdm_iterater = tqdm.tqdm(range(dataset.size()))
        else:
            tqdm_iterater = tqdm.tqdm(range(num_reqs))
        for _ in tqdm_iterater:
            messages, sampling_params = dataset.get()
            tokens = tokenizer.tokenize(messages_or_text=messages)

            prefix = self.radix_tree.match(tokens)
            self.prefix_hit_size.append((len(tokens), len(prefix)))
            self.radix_tree.insert(tokens)

        dataset.reset()

    def report(self) -> str:
        num_tokens = sum(x[0] for x in self.prefix_hit_size)
        hit_tokens = sum(x[1] for x in self.prefix_hit_size)

        dataset_prefix_hit_rate = hit_tokens / num_tokens
        request_prefix_hit_rate = numpy.mean(
            [x[1] / x[0] for x in self.prefix_hit_size]
        )

        num_requests = len(self.prefix_hit_size)

        avg_prompt_len = numpy.mean([x[0] for x in self.prefix_hit_size])

        return f"{num_requests=}, {num_tokens=}, {avg_prompt_len=}, {dataset_prefix_hit_rate=:.4f}, {request_prefix_hit_rate=:.4f}"


def main(cmd_arguments):
    tokenizer = Tokenizer(cmd_arguments.model)
    dataset = StdMessageJsonL(cmd_arguments.jsonl)
    profiler = DatasetProfiler(
        dataset,
        tokenizer,
        cmd_arguments.num_reqs,
        cmd_arguments.tk_capacity or sys.maxsize,
    )
    print(profiler.report())


if __name__ == "__main__":
    cmd_arguments_parser = argparse.ArgumentParser()
    cmd_arguments_parser.add_argument(
        "--model", type=str, help="path to model.", required=True
    )
    cmd_arguments_parser.add_argument(
        "--jsonl", type=str, help="path to jsonl dataset.", required=True
    )
    cmd_arguments_parser.add_argument(
        "--num-reqs",
        type=int,
        help="number of requests to profile, zero means depends on dataset.",
        default=0,
    )
    cmd_arguments_parser.add_argument(
        "--tk-capacity",
        type=int,
        help="radix tree size, zero means inf.",
        required=True,
    )
    cmd_arguments = cmd_arguments_parser.parse_args()

    main(cmd_arguments)
