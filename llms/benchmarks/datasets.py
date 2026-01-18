import abc
import sys
import json
import random
import logging

from typing import Any, Optional, Union, List, Dict, Tuple


Messages = List[Dict[str, str]]
SamplingParams = Dict[str, Any]


class Dataset:
    def __init__(self):
        self.index = 0
        self.messages_list: List[Messages] = []
        self.sampling_params_list: List[SamplingParams] = []

    @abc.abstractmethod
    def sleep_seconds(self) -> Optional[List[float]]:
        return

    def reset(self):
        self.index = 0

    def size(self) -> int:
        return len(self.messages_list)

    def get(self) -> Optional[Tuple[Messages, SamplingParams]]:
        if self.index >= len(self.messages_list):
            return None

        messages = self.messages_list[self.index]
        sampling_params = self.sampling_params_list[self.index]

        self.index += 1

        return messages, sampling_params


class StdMessageJsonL(Dataset):
    def __init__(self, dataset_path):
        super().__init__()

        with open(dataset_path, "r") as f:
            while line := f.readline():
                messages = json.loads(line)
                if isinstance(messages, dict):
                    messages = messages["messages"]

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

    def sleep_seconds(self):
        return None


class WeChatSearchQaOut2016V3300(Dataset):
    def __init__(
        self, dataset_path: str, set_random: bool, target_qps, qps_adjust_method: str
    ):
        super().__init__(set_random=set_random)

        self.system_prompts = []
        self.prompts = []
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
            for data in dataset:
                self.system_prompts.append(data["SystemPrompt"])
                self.prompts.append(data["Prompt"])

        self.num_messages = len(self.prompts)
        self.index = 0

        self._sleep_seconds = 1 / target_qps

    def sleep_seconds(self) -> Union[float, List[float]]:
        return self._sleep_seconds

    def get(self):
        system_prompt = self.system_prompts[self.index]
        prompt = self.prompts[self.index]
        self.index = (self.index + 1) % self.num_messages

        if self.set_random:
            random_str = str(random.randint(0, sys.maxsize)) + " "
        else:
            random_str = ""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": random_str + prompt},
        ], {}


class WeChatSearchQaOutFormatted(StdMessageJsonL):
    pass


class WeChatSearchDeepseekR1Fake0807(Dataset):
    # model_name,search_id,created_timestamp,prompt_token_num,completion_token_num
    # ...

    def __init__(self, dataset_path, set_random, target_qps, qps_adjust_method: str):
        super().__init__(set_random)

        import csv
        import ast

        def read_csv_to_dict_list(file_path: str):
            result = []
            with open(file_path, newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    result.append(dict(row))
            return result

        requests = read_csv_to_dict_list(dataset_path)
        requests = sorted(requests, key=lambda t: int(t["created_timestamp"]))

        self.messages_set = []
        self.sampling_params_set = []
        self.timestamp = []

        for request in requests:
            num_prompt_tokens = int(request["prompt_token_num"])
            num_output_tokens = int(request["completion_token_num"])

            self.messages_set.append(
                [{"role": "user", "content": "hi" * num_prompt_tokens}]
            )
            self.sampling_params_set.append(
                {
                    "max_tokens": num_output_tokens,
                    "extra_body": {"min_tokens": num_output_tokens},
                }
            )
            self.timestamp.append(int(request["created_timestamp"]) / 1000)  # ms -> s

        assert qps_adjust_method in ["sample", "stretch"]

        nseconds = self.timestamp[-1] - self.timestamp[0]
        nrequests = len(requests)
        dataset_qps = nrequests / nseconds
        logging.warning(f"{nrequests=}, {nseconds=}. {dataset_qps=}, {target_qps=}")

        if qps_adjust_method == "sample":
            sample_ratio = target_qps / dataset_qps
            sample_nrequests = int(sample_ratio * nrequests)
            random.seed(29)
            sample_indices = random.sample(range(nrequests), sample_nrequests)
            sample_indices.sort()

            self.messages_set = [self.messages_set[i] for i in sample_indices]
            self.sampling_params_set = [
                self.sampling_params_set[i] for i in sample_indices
            ]
            self.timestamp = [self.timestamp[i] for i in sample_indices]

            nseconds = self.timestamp[-1] - self.timestamp[0]
            nrequests = len(self.messages_set)
            qps = nrequests / nseconds
            logging.warning(
                f"adjust qps by [sample]: {nrequests=}, {nseconds=}, {qps=}"
            )

            self._sleep_seconds = [0] * len(self.timestamp)
            for i in range(0, len(self.timestamp) - 1):
                self._sleep_seconds[i] = self.timestamp[i + 1] - self.timestamp[i]

        else:
            self._sleep_seconds = [0] * len(self.timestamp)
            for i in range(0, len(self.timestamp) - 1):
                self._sleep_seconds[i] = self.timestamp[i + 1] - self.timestamp[i]

            stretch_ratio = dataset_qps / target_qps
            self._sleep_seconds = [x * stretch_ratio for x in self._sleep_seconds]

            nseconds = sum(self._sleep_seconds)
            nrequests = len(self.messages_set)
            qps = nrequests / nseconds
            logging.warning(
                f"adjust qps by [stretch]: {nrequests=}, {nseconds=}, {qps=}"
            )

        self.index = 0

    def sleep_seconds(self):
        return self._sleep_seconds

    def get(self):
        if self.index >= len(self.messages_set):
            return None

        messages = self.messages_set[self.index]
        sampling_params = self.sampling_params_set[self.index]
        self.index += 1

        return messages, sampling_params


class WeChatSearchQwQ32BConductorPlanCSV(Dataset):
    def __init__(self, dataset_path, set_random, target_qps, qps_adjust_method: str):
        super().__init__(set_random)

        import csv
        import ast

        logging.warning("WeChatSearchQwQ32BConductorPlanCSV do not support adjust qps.")

        def read_csv_to_dict_list(file_path: str):
            result = []
            with open(file_path, newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    result.append(dict(row))
            return result

        requests = read_csv_to_dict_list(dataset_path)

        self.messages_set = []
        self.sampling_params_set = []
        self._sleep_seconds = []
        for request in requests:
            if "http_body" in request:
                http_body = ast.literal_eval(request["http_body"])
                self.messages_set.append(http_body["messages"])
                self.sampling_params_set.append(
                    {
                        "max_tokens": http_body["max_tokens"],
                        "extra_body": {"p_rank": int(request["pnode_idx"])},
                    }
                )
            else:
                self.messages_set.append(
                    [{"role": "user", "content": "hi" * int(request["prompt_len"])}]
                )
                self.sampling_params_set.append(
                    {
                        "max_tokens": 400,
                        "extra_body": {
                            "min_tokens": 400,
                            "p_rank": int(request["pnode_idx"]),
                        },
                    }
                )

            self._sleep_seconds.append(float(request["offseted_timestamp"]))

        print("Dataset QPS:", len(self._sleep_seconds) / self._sleep_seconds[-1])

        for i in range(0, len(self._sleep_seconds) - 1):
            self._sleep_seconds[i] = self._sleep_seconds[i + 1] - self._sleep_seconds[i]
        self._sleep_seconds[-1] = 0

        self.index = 0

    def sleep_seconds(self):
        return self._sleep_seconds

    def get(self):
        if self.index >= len(self.messages_set):
            return None

        messages = self.messages_set[self.index]
        sampling_params = self.sampling_params_set[self.index]
        self.index += 1

        return messages, sampling_params
