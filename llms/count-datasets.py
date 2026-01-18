import argparse
import json

from transformers import AutoTokenizer


class Dataset:
    def __init__(self, dataset_path):
        self.dataset = []

        with open(dataset_path, "rb") as f:
            while messages := f.readline():
                messages = json.loads(messages)
                self.dataset.append(messages)


class Tokenizer:
    def __init__(self, tokenizer_path):
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )

    def tokenize(self, messages_or_text):
        if isinstance(messages_or_text, str):
            return self._tokenizer.encode(messages_or_text)
        else:
            return self._tokenizer.apply_chat_template(messages_or_text, tokenize=True)

    def count(self, messages_or_text):
        return len(self.tokenize(messages_or_text))

    def count_dataset(self, dataset: Dataset):
        num_tokens = 0
        for messages in dataset.dataset:
            num_tokens += self.count(messages)
        n = len(dataset.dataset)
        return num_tokens / n


if __name__ == "__main__":
    cmd_arguments = argparse.ArgumentParser()
    cmd_arguments.add_argument("--dataset", type=str, required=True)
    cmd_arguments.add_argument("--tokenizer", type=str, required=True)
    cmd_arguments = cmd_arguments.parse_args()

    avg_prompt_size = Tokenizer(cmd_arguments.tokenizer).count_dataset(
        Dataset(cmd_arguments.dataset)
    )
    print("Avg prompt size: %s" % avg_prompt_size)
