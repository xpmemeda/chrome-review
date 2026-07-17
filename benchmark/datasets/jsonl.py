import json
import logging
import typing as ty

from .base import Dataset, Messages, StdChatApiRequest


class JsonlTextDataset(Dataset):
    def __init__(self, dataset_path: str) -> None:
        self.dataset = self._load(dataset_path)
        if not self.dataset:
            raise RuntimeError(f"jsonl dataset is empty: {dataset_path}")
        logging.info(
            "loaded jsonl text dataset, path=%s, rows=%d",
            dataset_path,
            len(self.dataset),
        )

    def _load(self, dataset_path: str) -> ty.List[StdChatApiRequest]:
        dataset = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                messages = self._parse_messages(item, line_idx)
                dataset.append({"messages": messages})
        return dataset

    def _parse_messages(self, item: ty.Any, line_idx: int) -> Messages:
        if isinstance(item, str):
            return [{"role": "user", "content": item}]
        if isinstance(item, list):
            return self._normalize_messages(item, line_idx)
        if not isinstance(item, dict):
            raise RuntimeError(f"line {line_idx}: jsonl row must be object/list/string")
        if "messages" in item:
            return self._normalize_messages(item["messages"], line_idx)
        if "prompt" in item:
            messages = []
            if item.get("system_prompt"):
                messages.append(
                    {"role": "system", "content": str(item["system_prompt"])}
                )
            messages.append({"role": "user", "content": str(item["prompt"])})
            return messages
        if "role" in item and "content" in item:
            return self._normalize_messages([item], line_idx)
        raise RuntimeError(
            f"line {line_idx}: expected messages, prompt, a message object, or string"
        )

    def _normalize_messages(self, messages: ty.Any, line_idx: int) -> Messages:
        if not isinstance(messages, list):
            raise RuntimeError(f"line {line_idx}: messages must be a list")
        result = []
        for message_idx, message in enumerate(messages):
            if not isinstance(message, dict):
                raise RuntimeError(
                    f"line {line_idx}: message {message_idx} must be an object"
                )
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str):
                raise RuntimeError(
                    f"line {line_idx}: message {message_idx} role must be a string"
                )
            result.append(
                {
                    "role": role,
                    "content": self._normalize_text_content(
                        content,
                        line_idx,
                        message_idx,
                    ),
                }
            )
        if not result:
            raise RuntimeError(f"line {line_idx}: messages must not be empty")
        return result

    def _normalize_text_content(
        self,
        content: ty.Any,
        line_idx: int,
        message_idx: int,
    ) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part_idx, part in enumerate(content):
                if not isinstance(part, dict) or part.get("type") != "text":
                    raise RuntimeError(
                        f"line {line_idx}: message {message_idx} part {part_idx} "
                        "is not text-only"
                    )
                parts.append(str(part.get("text", "")))
            return "\n".join(parts)
        raise RuntimeError(
            f"line {line_idx}: message {message_idx} content must be text"
        )

    def get(self, req_idx: int) -> StdChatApiRequest:
        return self.dataset[req_idx % len(self.dataset)]

    def warmup(self, size: int) -> ty.List[StdChatApiRequest]:
        requests = []
        for idx in range(size):
            messages = self._warmup_prefix_messages()
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"warmup_unique_id: {idx}\n"
                        "This is a synthetic warmup request for benchmark setup."
                    ),
                }
            )
            requests.append({"messages": messages})
        return requests

    def _warmup_prefix_messages(self) -> Messages:
        request = self.dataset[0]
        messages = []
        messages.extend(
            {"role": message["role"], "content": message["content"]}
            for message in request["messages"]
            if message["role"] == "system"
        )
        messages.extend(
            {"role": message["role"], "content": message["content"]}
            for message in request["messages"]
            if message["role"] == "developer"
        )
        return messages

    def size(self) -> int:
        return len(self.dataset)
