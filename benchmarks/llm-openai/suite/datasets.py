import abc
import tqdm
import argparse
import json
import random
import typing as ty

Messages = ty.List[ty.Dict[str, str]]
SamplingParams = ty.Dict[str, ty.Any]


class Dataset:
    def __init__(self):
        self.index = 0
        self.messages_list: ty.List[Messages] = []
        self.sampling_params_list: ty.List[SamplingParams] = []

    @abc.abstractmethod
    def sleep_seconds(self) -> ty.Optional[ty.List[float]]:
        return None

    def reset(self):
        self.index = 0

    def size(self) -> int:
        return len(self.messages_list)

    def get(self) -> ty.Optional[ty.Tuple[Messages, SamplingParams]]:
        if self.index >= len(self.messages_list):
            return None

        messages = self.messages_list[self.index]
        sampling_params = self.sampling_params_list[self.index]

        self.index += 1

        return messages, sampling_params

    @classmethod
    @abc.abstractmethod
    def add_arguments(cls, parser: argparse.Namespace) -> None:
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def from_args(cls, args: argparse.Namespace) -> "Dataset":
        raise NotImplementedError()


class Synthetic(Dataset):
    def __init__(self, num_requests: int, prompt_length: int, prefix_hit_rate: int):
        super().__init__()

        random.seed(0)

        # 30 for templates: [{"role": "user", "content": ""}]
        # 10 for uuid
        prefix_size = int(prefix_hit_rate * prompt_length) - 40
        if prefix_size < 0:
            prefix_size = 0
        suffix_size = prompt_length - prefix_size - 40

        for i in tqdm.tqdm(range(num_requests)):
            prefix = "hi " * prefix_size
            uuid = "".join(random.choices("0123456789abcdef", k=16))
            uuid = uuid + " "
            suffix = "hi " * suffix_size

            messages = [{"role": "user", "content": prefix + uuid + suffix}]
            self.messages_list.append(messages)
            self.sampling_params_list.append({})

    @classmethod
    def add_arguments(cls, parser: argparse.Namespace) -> None:
        parser.add_argument("--ds-num-requests", type=int, required=True)
        parser.add_argument("--ds-prompt-length", type=int, required=True)
        parser.add_argument("--ds-prefix-hit-rate", type=int, required=True)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> Dataset:
        return cls(args.ds_num_requests, args.ds_prompt_length, args.ds_prefix_hit_rate)


class StdMessageJsonL(Dataset):
    def __init__(self, dataset_path):
        super().__init__()

        with open(dataset_path, "r") as f:
            while line := f.readline():
                messages = json.loads(line)

                # raw string.
                if isinstance(messages, str):
                    self.messages_list.append([{"role": "user", "content": messages}])
                    self.sampling_params_list.append({})
                    continue

                # std messages.
                if isinstance(messages, dict):
                    messages = messages["messages"]

                # multi-part messages.
                """ OpenAI-style multi-part message
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hello"},
                            ...
                        ]
                    }
                """
                for message in messages:
                    content = message["content"]
                    if isinstance(content, list):
                        if len(content) != 1 or "text" not in content[0]:
                            raise RuntimeError(
                                "Maybe multimodal datasets, can't count."
                            )
                        message["content"] = content[0]["text"]

                self.messages_list.append(messages)

                self.sampling_params_list.append({})

    @classmethod
    def add_arguments(cls, parser: argparse.Namespace) -> None:
        parser.add_argument("--ds-dataset-path", type=str, required=True)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> Dataset:
        return cls(args.dataset_path)


def get_cls(dataset_type) -> ty.Type[Dataset]:
    if dataset_type == "synthetic":
        return Synthetic
    elif dataset_type == "jsonl":
        return StdMessageJsonL
    else:
        raise RuntimeError(f"unknown dataset type: {dataset_type}")
