import base64
import logging
import random
import sys
import typing as ty

import tqdm

from .base import StdChatApiRequest, VlmDataset
from .synthetic_utils import (
    check_prompt_prefix_hit_rate,
    encode_png_rgb,
    make_base_rgb_rows,
    make_synthetic_prompt,
    make_synthetic_system_prompt,
    pad_png_to_size,
)

IMAGE_TARGET_KIB = 32


class SyntheticVlmDataset(VlmDataset):
    def __init__(
        self,
        num_requests: int,
        num_prompt_tokens: int,
        prompt_prefix_hit_rate: float,
        image_width: int,
        image_height: int,
        image_seed: ty.Optional[int],
    ) -> None:
        if (
            not num_requests
            or not num_prompt_tokens
            or not image_width
            or not image_height
        ):
            raise RuntimeError("Please provide arguments for SyntheticVlmDataset.")
        if image_seed is None:
            image_seed = random.randint(0, sys.maxsize)
        check_prompt_prefix_hit_rate(prompt_prefix_hit_rate)

        self.num_prompt_tokens = num_prompt_tokens
        self.num_prompt_prefix_tokens = int(num_prompt_tokens * prompt_prefix_hit_rate)
        self.num_prompt_suffix_tokens = (
            num_prompt_tokens - self.num_prompt_prefix_tokens
        )
        self.image_width = image_width
        self.image_height = image_height
        self.image_seed = image_seed
        self.image_target_bytes = IMAGE_TARGET_KIB * 1024
        self.system_prompt = make_synthetic_system_prompt(self.num_prompt_prefix_tokens)
        self.base_rows = make_base_rgb_rows(
            self.image_width,
            self.image_height,
            image_seed,
        )

        self.dataset: ty.List[StdChatApiRequest] = []

        logging.info(
            "preparing synthetic vlm dataset ing, seed=%d, image_width=%d, "
            "image_height=%d, image_target_kib=%d, prompt_prefix_hit_rate=%.4f",
            self.image_seed,
            self.image_width,
            self.image_height,
            IMAGE_TARGET_KIB,
            prompt_prefix_hit_rate,
        )
        self._prepare(num_requests)
        logging.info("preparing synthetic vlm dataset done.")

    def _make_png(self, req_idx: int) -> bytes:
        rows = bytearray(self.base_rows)
        self._patch_rows(rows, req_idx)
        png = encode_png_rgb(self.image_width, self.image_height, rows)
        return pad_png_to_size(png, self.image_target_bytes, self.image_seed + req_idx)

    def _patch_rows(self, rows: bytearray, req_idx: int) -> None:
        rng = random.Random(self.image_seed + req_idx)
        num_pixels = min(16, self.image_width * self.image_height)
        for patch_idx in range(num_pixels):
            pos = rng.randrange(self.image_width * self.image_height)
            y, x = divmod(pos, self.image_width)
            offset = y * (1 + self.image_width * 3) + 1 + x * 3
            value = (req_idx * 131 + patch_idx * 17) & 0xFF
            rows[offset] = (rows[offset] + value + 1) & 0xFF
            rows[offset + 1] = (rows[offset + 1] ^ (value + 97)) & 0xFF
            rows[offset + 2] = (rows[offset + 2] + value + 193) & 0xFF

    def _prepare(self, num_requests) -> None:
        tqdm_iter = tqdm.tqdm(range(num_requests))
        for idx in tqdm_iter:
            prompt = make_synthetic_prompt(
                idx,
                self.image_seed,
                self.num_prompt_suffix_tokens,
            )
            self.dataset.append(
                {"messages": self._make_messages(prompt, self._make_png(idx))}
            )

    def get(self, req_idx: int) -> StdChatApiRequest:
        return self.dataset[req_idx]

    def warmup(self, size: int) -> ty.List[StdChatApiRequest]:
        requests = []
        for idx in range(size):
            warmup_idx = len(self.dataset) + idx
            prompt = make_synthetic_prompt(
                warmup_idx,
                self.image_seed,
                self.num_prompt_suffix_tokens,
            )
            requests.append(
                {
                    "messages": self._make_messages(
                        f"warmup_unique_id: {idx}\n{prompt}",
                        self._make_png(warmup_idx),
                    )
                }
            )
        return requests

    def _make_messages(
        self,
        prompt: str,
        image_bytes: bytes,
    ) -> ty.List[ty.Dict[str, ty.Any]]:
        image_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode(
            "ascii"
        )
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url, "detail": "low"},
                    },
                ],
            }
        )
        return messages

    def size(self) -> int:
        return len(self.dataset)
