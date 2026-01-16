from transformers import AutoTokenizer
from typing import List


class Tokenizer:
    def __init__(self, tokenizer_path):
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )

    def tokenize(self, messages_or_text) -> List[int]:
        if isinstance(messages_or_text, str):
            tokenize_result = self._tokenizer.encode(messages_or_text)
        else:
            tokenize_result = self._tokenizer.apply_chat_template(
                messages_or_text, tokenize=True
            )

        if isinstance(tokenize_result, list):
            return tokenize_result

        return tokenize_result["input_ids"]

    def count(self, messages_or_text) -> int:
        return len(self.tokenize(messages_or_text))
