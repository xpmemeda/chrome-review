import typing as ty

from .base import TokenCounter


class TransformersTokenCounter(TokenCounter):
    def __init__(self, tokenizer_path: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "Transformers token counting requires the transformers Python package."
            ) from e

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )

    def tokenize(self, messages_or_text: ty.Any) -> ty.List[int]:
        if isinstance(messages_or_text, str):
            return self.tokenizer.encode(messages_or_text)
        return self.tokenizer.apply_chat_template(messages_or_text, tokenize=True)
