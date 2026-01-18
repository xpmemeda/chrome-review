import numpy
import tqdm

from typing import List, Tuple

from datasets import Dataset
from tokenizer import Tokenizer
from comm.radixtree import RadixTree


class DatasetProfiler:
    def __init__(self, dataset: Dataset, tokenizer: Tokenizer, radix_size: int):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.radix_tree = RadixTree(radix_size)

        self.prefix_hit_size: List[Tuple[int, int]] = []

        tqdm_iterater = tqdm.tqdm(range(dataset.size()))
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

        num_requests = self.dataset.size()

        return f"{num_requests=}, {num_tokens=}, {dataset_prefix_hit_rate=:.4f}, {request_prefix_hit_rate=:.4f}"
