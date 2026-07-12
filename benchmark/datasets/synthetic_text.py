import logging
import typing as ty

import tqdm

from .base import StdChatApiRequest, VlmDataset
from .synthetic_utils import (
    check_prompt_prefix_hit_rate,
    make_synthetic_prompt,
    make_synthetic_system_prompt,
)


class SyntheticTextDataset(VlmDataset):
    def __init__(
        self,
        num_requests: int,
        num_prompt_tokens: int,
        prompt_prefix_hit_rate: float,
        seed: int,
    ) -> None:
        if not num_requests or not num_prompt_tokens:
            raise RuntimeError("Please provide arguments for SyntheticTextDataset.")
        if seed is None:
            seed = 0
        check_prompt_prefix_hit_rate(prompt_prefix_hit_rate)

        self.num_prompt_tokens = num_prompt_tokens
        self.num_prompt_prefix_tokens = int(num_prompt_tokens * prompt_prefix_hit_rate)
        self.num_prompt_suffix_tokens = (
            num_prompt_tokens - self.num_prompt_prefix_tokens
        )
        self.seed = seed
        self.system_prompt = make_synthetic_system_prompt(self.num_prompt_prefix_tokens)
        self.dataset: ty.List[StdChatApiRequest] = []

        logging.info(
            "preparing synthetic text dataset ing, seed=%d, prompt_prefix_hit_rate=%.4f",
            self.seed,
            prompt_prefix_hit_rate,
        )
        self._prepare(num_requests)
        logging.info("preparing synthetic text dataset done.")

    def _prepare(self, num_requests: int) -> None:
        tqdm_iter = tqdm.tqdm(range(num_requests))
        for idx in tqdm_iter:
            prompt = make_synthetic_prompt(
                idx,
                self.seed,
                self.num_prompt_suffix_tokens,
            )
            self.dataset.append({"messages": self._make_messages(prompt)})

    def get(self, req_idx: int) -> StdChatApiRequest:
        return self.dataset[req_idx]

    def warmup(self, size: int) -> ty.List[StdChatApiRequest]:
        requests = []
        for idx in range(size):
            prompt = make_synthetic_prompt(
                idx,
                self.seed - 100000000,
                self.num_prompt_suffix_tokens,
            )
            requests.append(
                {"messages": self._make_messages(f"warmup_unique_id: {idx}\n{prompt}")}
            )
        return requests

    def _make_messages(self, prompt: str) -> ty.List[ty.Dict[str, ty.Any]]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def size(self) -> int:
        return len(self.dataset)
